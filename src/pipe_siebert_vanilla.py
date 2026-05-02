"""
Siebert 5-fold + OOF 收集
==========================
和 Day 1 的 Sentiment pipeline.py 完全一樣的訓練流程,
但額外儲存:
  - oof_siebert.npy : shape (2000, 2), 每筆 train 樣本被「fold 沒看過它的模型」預測的機率
  - test_probs_siebert.npy : shape (10999, 2), 5-fold 平均 test 機率 (= 原本的 avg_probs.npy)

OOF 是 stacking 的關鍵:
  - meta-learner 的 train 必須用 OOF 才不會 leak
  - 5-fold CV 中 fold-i 沒看過 fold-i 的 val, 所以 fold-i 的 val 預測就是這些樣本的 OOF
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
FINETUNE_DIR = f"{OUTPUT_DIR}/folds_siebert_vanilla"
OOF_FILE     = f"{OUTPUT_DIR}/oof_siebert_vanilla.npy"
TEST_FILE    = f"{OUTPUT_DIR}/testprobs_siebert_vanilla.npy"

BASE_MODEL   = "siebert/sentiment-roberta-large-english"
MAX_LENGTH   = 64
SEED         = 42
N_FOLDS      = 5
FT_EPOCHS    = 3
FT_BATCH_SIZE  = 8
FT_EVAL_BATCH  = 16
FT_LR        = 1e-5

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
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    texts  = train_df[TEXT_COL].tolist()
    labels = train_df[LABEL_COL].tolist()
    test_texts = test_df[TEXT_COL].tolist()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probs   = np.zeros((len(texts), 2), dtype=np.float32)
    test_probs  = np.zeros((len(test_texts), 2), dtype=np.float32)
    fold_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        tr_texts  = [texts[i]  for i in tr_idx]
        tr_labels = [labels[i] for i in tr_idx]
        va_texts  = [texts[i]  for i in va_idx]
        va_labels = [labels[i] for i in va_idx]

        fold_dir = f"{FINETUNE_DIR}/fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True,
        )
        tr_ds = SentimentDataset(tr_texts, tr_labels, tokenizer, MAX_LENGTH)
        va_ds = SentimentDataset(va_texts, va_labels, tokenizer, MAX_LENGTH)

        args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=FT_EPOCHS,
            per_device_train_batch_size=FT_BATCH_SIZE,
            per_device_eval_batch_size=FT_EVAL_BATCH,
            learning_rate=FT_LR,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=20,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            seed=SEED + fold_idx,
            report_to="none",
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=tr_ds, eval_dataset=va_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )
        trainer.train()
        val_res = trainer.evaluate()
        fold_scores.append(val_res["eval_f1"])
        print(f"Fold {fold_idx + 1} Val F1 = {val_res['eval_f1']:.4f}")

        # 關鍵: 存 val 預測作為這部分樣本的 OOF
        val_probs = predict_probs(trainer.model, tokenizer, va_texts)
        oof_probs[va_idx] = val_probs

        # 同時累積 test 預測 (5-fold 平均)
        test_probs += predict_probs(trainer.model, tokenizer, test_texts) / N_FOLDS

        del trainer, model
        torch.cuda.empty_cache()

    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))
    print("\n" + "="*50)
    print(f"Siebert CV F1 : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Day 1 baseline: 0.8710 ± 0.0052")

    # OOF F1 是另一個健康度檢查 (應該接近 fold mean)
    oof_preds = np.argmax(oof_probs, axis=1)
    oof_f1 = f1_score(labels, oof_preds, average="macro")
    print(f"OOF F1        : {oof_f1:.4f}  (sanity: 應接近 CV mean)")

    np.save(OOF_FILE, oof_probs)
    np.save(TEST_FILE, test_probs)
    print(f"\nOOF probs saved : {OOF_FILE}")
    print(f"Test probs saved: {TEST_FILE}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
