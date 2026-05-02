"""
Siebert Soft Pseudo-Labeling
=============================

跟之前 hard pseudo (conf >= 0.95) 失敗的差別:
  - Hard 版只取 confident test 樣本當 one-hot label,本質是「加強自己原本就確定的決策」
  - Soft 版保留軟分布 p_test (來自 teacher 的 5-fold 平均 test probs),
    用 soft-CE / KL 訓練,模型可以表達「這筆其實 0.6/0.4 不確定」

流程:
  Stage 1 (teacher):
    - 直接讀 outputs/testprobs_siebert_vanilla.npy 當 teacher soft labels
    - (vanilla 是目前 LB 最強的 baseline,先用它當 teacher;之後可換成
       multi-seed-clean 或 siebert+cardiff+electra 平均做多樣性版本)

  Stage 2 (student soft pseudo fine-tune):
    - 訓練資料 = 真實 train (2000, hard one-hot) + test (N_test, soft p_test)
    - 可選 confidence filter τ:只保留 max(p_test) >= τ 的 pseudo 樣本
    - Loss = CE(real) + α · soft_CE(pseudo)
    - 5-fold split 只切真實 2000 筆,pseudo 樣本永遠進 train side、
      val 永遠是純真實標籤

  Stage 3 (threshold tuning):
    - 走 tool_threshold-tune-foldwise.py 做 per-fold best_t 收斂判定

輸出:
  - oof_siebert_soft-pseudo_<tag>.npy        : (2000, 2)
  - oof_siebert_soft-pseudo_<tag>_per-fold.npy : (5, 2000, 2) 只有 val 位置非零
  - testprobs_siebert_soft-pseudo_<tag>.npy  : (10999, 2)

CLI 範例:
  python pipe_siebert_soft-pseudo.py --tau 0.8 --alpha 0.5
  python pipe_siebert_soft-pseudo.py --tau 0.0 --alpha 1.0   # 不做 confidence filter
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

TRAIN_CSV    = "data/train_2022.csv"
TEST_CSV     = "data/test_no_answer_2022.csv"
TEXT_COL  = "TEXT"
LABEL_COL = "LABEL"
ID_COL    = "row_id"

OUTPUT_DIR   = "./outputs"
TEACHER_FILE = f"{OUTPUT_DIR}/testprobs_siebert_vanilla.npy"
FINETUNE_DIR = f"{OUTPUT_DIR}/folds_siebert_soft-pseudo"

BASE_MODEL  = "siebert/sentiment-roberta-large-english"
MAX_LENGTH  = 64
N_FOLDS     = 5
SEED        = 42
FT_EPOCHS   = 3
FT_BATCH    = 8
FT_EVAL_BATCH = 16
FT_LR       = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MixedDataset(Dataset):
    """混合資料集: 真實樣本 (hard one-hot) + pseudo 樣本 (soft prob)。

    每筆固定包含: input_ids, attention_mask, labels (hard int, eval 用),
    soft_labels (shape=[2], loss 用), is_real (1=真實, 0=pseudo)。
    """
    def __init__(self, encodings, soft_labels, hard_labels, is_real, indices):
        self.encodings = encodings
        self.soft_labels = soft_labels
        self.hard_labels = hard_labels
        self.is_real = is_real
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["soft_labels"] = self.soft_labels[idx]
        item["is_real"] = self.is_real[idx]
        item["labels"] = self.hard_labels[idx]
        return item


class SoftCETrainer(Trainer):
    """Loss: real 走 hard CE, pseudo 走 soft CE,以 alpha 加權混合。"""

    def __init__(self, *args, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft_labels = inputs.pop("soft_labels")  # [B, 2]
        is_real = inputs.pop("is_real").bool()    # [B]
        inputs.pop("labels", None)                # 不讓 model 內部重複算 CE
        outputs = model(**inputs)
        logits = outputs.logits                   # [B, 2]
        log_probs = torch.log_softmax(logits, dim=-1)

        # soft CE = - Σ p · log q  (對 real 樣本來說 p 是 one-hot,等價於 hard CE)
        per_sample = -(soft_labels * log_probs).sum(dim=-1)   # [B]

        n_real = is_real.sum().clamp(min=1)
        n_pseudo = (~is_real).sum().clamp(min=1)
        loss_real = (per_sample * is_real.float()).sum() / n_real
        loss_pseudo = (per_sample * (~is_real).float()).sum() / n_pseudo

        # 若 batch 沒有 pseudo,只算 real;反之亦然
        has_real = is_real.any().float()
        has_pseudo = (~is_real).any().float()
        loss = has_real * loss_real + self.alpha * has_pseudo * loss_pseudo

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # eval_dataset 只放真實樣本,labels 已是 hard 整數
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


class EvalRealDataset(Dataset):
    """Eval 專用: 只放真實樣本,但介面跟 MixedDataset 一致(都帶
    soft_labels + is_real),這樣 SoftCETrainer.compute_loss 不會在
    eval 時找不到欄位。is_real 全 1 → loss 自動退回純 hard CE。"""
    def __init__(self, encodings, labels, indices):
        self.encodings = encodings
        self.labels = labels
        self.indices = indices
        # one-hot 軟標 (eval 時 loss 會走 real 分支,值正確即可)
        self._soft = torch.zeros(len(labels), 2, dtype=torch.float32)
        self._soft[torch.arange(len(labels)), labels] = 1.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["soft_labels"] = self._soft[idx]
        item["is_real"] = torch.tensor(1, dtype=torch.long)
        return item


def predict_probs(model, encodings, indices=None, batch_size=FT_EVAL_BATCH):
    model.to(device)
    model.eval()
    if indices is None:
        indices = np.arange(len(encodings["input_ids"]))
    out = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            bi = indices[i:i + batch_size]
            input_ids = encodings["input_ids"][bi].to(device)
            attention_mask = encodings["attention_mask"][bi].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.8,
                    help="Confidence filter on teacher: keep test sample if max(p_test) >= tau. 0 = keep all.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight on pseudo loss term.")
    ap.add_argument("--teacher", type=str, default=TEACHER_FILE,
                    help="Teacher test probs file (.npy, shape [N_test, 2]).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Output tag suffix; default auto = tau{τ}_a{α}")
    args = ap.parse_args()

    if args.tag is None:
        args.tag = f"tau{args.tau:.2f}_a{args.alpha:.2f}".replace(".", "")

    oof_file       = f"{OUTPUT_DIR}/oof_siebert_soft-pseudo_{args.tag}.npy"
    oof_perfold    = f"{OUTPUT_DIR}/oof_siebert_soft-pseudo_{args.tag}_per-fold.npy"
    test_file      = f"{OUTPUT_DIR}/testprobs_siebert_soft-pseudo_{args.tag}.npy"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINETUNE_DIR, exist_ok=True)
    set_seed(SEED)

    print(f"Device: {device}")
    print(f"tau={args.tau} | alpha={args.alpha} | teacher={args.teacher}")
    print(f"Output tag: {args.tag}")

    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    n_train = len(train_df)
    n_test  = len(test_df)
    print(f"Train: {n_train} | Test: {n_test}")

    teacher = np.load(args.teacher).astype(np.float32)
    assert teacher.shape == (n_test, 2), f"Teacher shape {teacher.shape} != ({n_test}, 2)"
    teacher_max = teacher.max(axis=1)
    print(f"Teacher P(pos) mean = {teacher[:, 1].mean():.4f}")
    print(f"Teacher max-conf mean = {teacher_max.mean():.4f}, min = {teacher_max.min():.4f}")

    # Confidence filter
    if args.tau > 0:
        keep_mask = teacher_max >= args.tau
        n_keep = int(keep_mask.sum())
        print(f"Confidence filter τ={args.tau}: keep {n_keep}/{n_test} ({n_keep/n_test*100:.1f}%) test samples")
    else:
        keep_mask = np.ones(n_test, dtype=bool)
        n_keep = n_test
        print(f"τ=0: 保留全部 {n_test} 筆 test 當 pseudo")

    # Tokenize 一次
    print("\nTokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_enc = tokenizer(train_df[TEXT_COL].tolist(), truncation=True,
                          padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
    test_enc = tokenizer(test_df[TEXT_COL].tolist(), truncation=True,
                         padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")

    train_labels_int = torch.tensor(train_df[LABEL_COL].values, dtype=torch.long)

    # 建立合併 encoding: 前 n_train 是真實, 後面 n_keep 是 kept pseudo
    keep_idx = np.where(keep_mask)[0]
    pseudo_enc = {k: v[keep_idx] for k, v in test_enc.items()}

    combined_enc = {
        k: torch.cat([train_enc[k], pseudo_enc[k]], dim=0)
        for k in train_enc.keys()
    }

    # soft labels for combined: real 是 one-hot(從 hard label 來), pseudo 是 teacher prob
    real_soft = torch.zeros(n_train, 2, dtype=torch.float32)
    real_soft[torch.arange(n_train), train_labels_int] = 1.0
    pseudo_soft = torch.tensor(teacher[keep_idx], dtype=torch.float32)
    combined_soft = torch.cat([real_soft, pseudo_soft], dim=0)

    is_real = torch.cat([
        torch.ones(n_train, dtype=torch.long),
        torch.zeros(n_keep, dtype=torch.long),
    ])

    # combined hard labels: real 用真實 label, pseudo 用 argmax(teacher) (僅 eval/欄位齊一用)
    pseudo_hard = torch.tensor(teacher[keep_idx].argmax(axis=1), dtype=torch.long)
    combined_hard = torch.cat([train_labels_int, pseudo_hard], dim=0)

    # Eval (val) 用 hard labels: 仍指 train_enc
    train_labels = train_labels_int  # alias

    # 5-fold split 只在真實 train 上
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pseudo_indices = np.arange(n_train, n_train + n_keep)  # 在 combined_enc 中的 index

    oof_probs   = np.zeros((n_train, 2), dtype=np.float32)
    oof_perfold_arr = np.zeros((N_FOLDS, n_train, 2), dtype=np.float32)
    test_probs  = np.zeros((n_test, 2), dtype=np.float32)
    fold_scores = []
    fold_best_t = []  # 留給後續 foldwise threshold 工具用

    for fold_idx, (tr_idx_real, va_idx_real) in enumerate(
        skf.split(np.arange(n_train), train_labels.numpy())
    ):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        fold_dir = f"{FINETUNE_DIR}/fold{fold_idx}_{args.tag}"
        os.makedirs(fold_dir, exist_ok=True)

        # train indices in combined_enc: real-train 部分 + 全部 pseudo
        tr_idx = np.concatenate([tr_idx_real, pseudo_indices]).astype(np.int64)
        va_idx = va_idx_real.astype(np.int64)

        tr_ds = MixedDataset(combined_enc, combined_soft, combined_hard, is_real, tr_idx)
        va_ds = EvalRealDataset(train_enc, train_labels, va_idx)

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True,
        )

        targs = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=FT_EPOCHS,
            per_device_train_batch_size=FT_BATCH,
            per_device_eval_batch_size=FT_EVAL_BATCH,
            learning_rate=FT_LR,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            seed=SEED * 100 + fold_idx,
            data_seed=SEED * 100 + fold_idx,
            report_to="none",
            remove_unused_columns=False,  # 必須關掉,否則 soft_labels/is_real 被砍掉
        )

        trainer = SoftCETrainer(
            model=model, args=targs,
            train_dataset=tr_ds, eval_dataset=va_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
            alpha=args.alpha,
        )
        trainer.train()
        val_res = trainer.evaluate()
        fold_scores.append(val_res["eval_f1"])
        print(f"Fold {fold_idx + 1} Val F1 = {val_res['eval_f1']:.4f}")

        val_probs = predict_probs(trainer.model, train_enc, va_idx)
        oof_probs[va_idx] = val_probs
        oof_perfold_arr[fold_idx, va_idx] = val_probs

        test_probs += predict_probs(trainer.model, test_enc) / N_FOLDS

        # per-fold best_t (在這個 fold 的 val 上)
        y_va = train_labels[va_idx].numpy()
        ts = np.arange(0.20, 0.80 + 1e-9, 0.005)
        f1s = [f1_score(y_va, (val_probs[:, 1] >= t).astype(int), average="macro") for t in ts]
        bt = float(ts[int(np.argmax(f1s))])
        bf = float(np.max(f1s))
        fold_best_t.append((bt, bf))
        print(f"Fold {fold_idx + 1} best_t = {bt:.3f} (val F1 = {bf:.4f})")

        del trainer, model
        torch.cuda.empty_cache()
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)

    # Summary
    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))
    y = train_df[LABEL_COL].values.astype(int)
    f1_oof = f1_score(y, oof_probs.argmax(1), average="macro")

    print("\n" + "=" * 60)
    print(f"SOFT PSEUDO SUMMARY  (tag={args.tag})")
    print("=" * 60)
    for i, (bt, bf) in enumerate(fold_best_t):
        print(f"Fold {i+1}: val F1@0.5 = {fold_scores[i]:.4f}  | best_t = {bt:.3f}  best_F1 = {bf:.4f}")
    bts = [b for b, _ in fold_best_t]
    print(f"per-fold best_t: {bts}  spread = {max(bts) - min(bts):.3f}")
    print(f"CV F1 (argmax): {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"OOF F1 (argmax): {f1_oof:.4f}")
    print(f"vanilla baseline OOF F1 = 0.8710")

    np.save(oof_file, oof_probs)
    np.save(oof_perfold, oof_perfold_arr)
    np.save(test_file, test_probs)
    print(f"\nSaved:")
    print(f"  {oof_file}")
    print(f"  {oof_perfold}")
    print(f"  {test_file}")


if __name__ == "__main__":
    main()
