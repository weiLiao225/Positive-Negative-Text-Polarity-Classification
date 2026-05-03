"""
google/electra-large-discriminator Stratified 5-fold fine-tune。
fold split 與 pipe_siebert_vanilla.py 相同(seed=42),OOF 可直接對齊做 ensemble。
val F1 < 0.60 時自動換 seed 重訓,最多 2 次。

Outputs:
  outputs/oof_electra_vanilla.npy       : (2000, 2) OOF 機率
  outputs/testprobs_electra_vanilla.npy : (10999, 2) 5-fold 平均 test 機率

Usage:
  python src/pipe_electra_vanilla.py
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

TRAIN_CSV    = "data/train_2022.csv"
TEST_CSV     = "data/test_no_answer_2022.csv"
TEXT_COL     = "TEXT"
LABEL_COL    = "LABEL"
ID_COL       = "row_id"

OUTPUT_DIR   = "./outputs"
FINETUNE_DIR = f"{OUTPUT_DIR}/folds_electra_vanilla"
OOF_FILE     = f"{OUTPUT_DIR}/oof_electra_vanilla.npy"
TEST_FILE    = f"{OUTPUT_DIR}/testprobs_electra_vanilla.npy"

BASE_MODEL   = "google/electra-large-discriminator"
MAX_LENGTH   = 64
SEED         = 42       # 必須與 pipe_siebert_vanilla.py 一致,fold split 才能對齊
N_FOLDS      = 5
FT_EPOCHS    = 5
FT_BATCH_SIZE  = 8
FT_EVAL_BATCH  = 16
FT_LR        = 5e-6     # ELECTRA-large 在小資料對 lr 敏感,使用較小值
FT_WARMUP    = 0.2
FT_PATIENCE  = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINETUNE_DIR, exist_ok=True)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class PredictDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def predict_probs(model, tokenizer, texts):
    ds = PredictDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=FT_EVAL_BATCH, shuffle=False)
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    texts  = train_df[TEXT_COL].tolist()
    labels = train_df[LABEL_COL].tolist()
    test_texts = test_df[TEXT_COL].tolist()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probs   = np.zeros((len(texts), 2), dtype=np.float32)
    test_probs  = np.zeros((len(test_texts), 2), dtype=np.float32)
    fold_scores = []

    COLLAPSE_F1   = 0.60   # F1 < 此值判為 collapse
    MAX_RETRIES   = 2

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        tr_texts  = [texts[i]  for i in tr_idx]
        tr_labels = [labels[i] for i in tr_idx]
        va_texts  = [texts[i]  for i in va_idx]
        va_labels = [labels[i] for i in va_idx]

        fold_dir = f"{FINETUNE_DIR}/fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)

        tr_ds = SentimentDataset(tr_texts, tr_labels, tokenizer, MAX_LENGTH)
        va_ds = SentimentDataset(va_texts, va_labels, tokenizer, MAX_LENGTH)

        attempt = 0
        while True:
            seed_for_run = SEED + fold_idx + attempt * 100   # retry 換 seed
            print(f"  attempt {attempt + 1} (seed={seed_for_run})")

            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, num_labels=2,
            )

            args = TrainingArguments(
                output_dir=fold_dir,
                num_train_epochs=FT_EPOCHS,
                per_device_train_batch_size=FT_BATCH_SIZE,
                per_device_eval_batch_size=FT_EVAL_BATCH,
                learning_rate=FT_LR,
                warmup_ratio=FT_WARMUP,
                weight_decay=0.01,
                adam_epsilon=1e-6,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                logging_steps=20,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                seed=seed_for_run,
                report_to="none",
            )

            trainer = Trainer(
                model=model, args=args,
                train_dataset=tr_ds, eval_dataset=va_ds,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=FT_PATIENCE)],
            )
            trainer.train()
            val_res = trainer.evaluate()
            f1 = val_res["eval_f1"]
            print(f"  Fold {fold_idx + 1} attempt {attempt + 1} Val F1 = {f1:.4f}")

            if f1 >= COLLAPSE_F1 or attempt >= MAX_RETRIES:
                break
            print(f"  ⚠ collapse detected (F1 < {COLLAPSE_F1}), retrying with new seed...")
            del trainer, model
            torch.cuda.empty_cache()
            attempt += 1

        if f1 < COLLAPSE_F1:
            print(f"  ✗ Fold {fold_idx + 1} 重試 {MAX_RETRIES} 次仍 collapse, 放棄此 fold")
        fold_scores.append(f1)

        val_probs = predict_probs(trainer.model, tokenizer, va_texts)
        oof_probs[va_idx] = val_probs

        test_probs += predict_probs(trainer.model, tokenizer, test_texts) / N_FOLDS

        del trainer, model
        torch.cuda.empty_cache()

    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))
    print("\n" + "="*50)
    print(f"ELECTRA-large CV F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")

    oof_preds = np.argmax(oof_probs, axis=1)
    oof_f1 = f1_score(labels, oof_preds, average="macro")
    print(f"OOF F1             : {oof_f1:.4f}")

    siebert_oof_path = f"{OUTPUT_DIR}/oof_siebert_vanilla.npy"
    if os.path.exists(siebert_oof_path):
        siebert_oof = np.load(siebert_oof_path)
        agree = (siebert_oof.argmax(1) == oof_preds).mean()
        print(f"\nELECTRA vs siebert OOF 一致率: {agree*100:.2f}%  (85~92% 為 ensemble 有效區間)")

    np.save(OOF_FILE, oof_probs)
    np.save(TEST_FILE, test_probs)
    print(f"\nOOF probs saved : {OOF_FILE}")
    print(f"Test probs saved: {TEST_FILE}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
