"""
Foldwise Threshold Tuning
=========================

判定 OOF threshold 是否為「真訊號」而非全 OOF 上的偽峰。

Day 5 結論:全 OOF 上掃出的 best_t (例如 multi-seed Mosbach 的 0.62) 可能是
不同 seed/fold 預測抵銷後的 mode 重疊雜訊。判別方式:對每個 fold 各自掃
best_t,看 5 個 fold 的 best_t 是否收斂(spread < 0.10)。

收斂判定:
  - spread = max(best_t_per_fold) - min(best_t_per_fold)
  - spread < 0.10 → 真訊號,採 fold-median best_t
  - spread >= 0.10 → 噪聲,退回 t=0.5

輸入:
  - oof_<tag>.npy            (2000, 2)     OOF probs (argmax 後 = OOF prediction)
  - oof_<tag>_per-fold.npy   (n_folds, 2000, 2)  每 fold 只在 val 位置非零
  - testprobs_<tag>.npy      (n_test, 2)   test probs(用最終 threshold 套上去)

用法:
  python tool_threshold-tune-foldwise.py --tag soft-pseudo_tau080_a050
  python tool_threshold-tune-foldwise.py --tag soft-pseudo_tau080_a050 --write-submission
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

TRAIN_CSV    = "data/train_2022.csv"
TEST_CSV     = "data/test_no_answer_2022.csv"
OUT_DIR   = "outputs"

CONVERGENCE_SPREAD = 0.15   # spread 小於這個算(弱)收斂; Day 5 multi-seed 偽訊號 spread = 0.21~0.44 屬於發散


def sweep_one(probs_pos, y_true, lo=0.20, hi=0.80, step=0.005):
    ts = np.arange(lo, hi + 1e-9, step)
    f1s = np.array([
        f1_score(y_true, (probs_pos >= t).astype(int), average="macro") for t in ts
    ])
    bi = int(np.argmax(f1s))
    return ts, f1s, float(ts[bi]), float(f1s[bi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="模型 tag,對應 oof_siebert_soft-pseudo_<tag>.npy 等檔。"
                         "或直接給 'siebert_vanilla' 等其他模型 tag。")
    ap.add_argument("--prefix", default="siebert_soft-pseudo",
                    help="檔名前綴,預設 siebert_soft-pseudo (對應 pipe_siebert_soft-pseudo.py 輸出)")
    ap.add_argument("--range", type=float, nargs=2, default=[0.20, 0.80])
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--write-submission", action="store_true")
    ap.add_argument("--force-t", type=float, default=None,
                    help="強制使用此 threshold,跳過收斂判定(用於明確要交某個 t 的 submission)")
    args = ap.parse_args()

    oof_path     = os.path.join(OUT_DIR, f"oof_{args.prefix}_{args.tag}.npy")
    perfold_path = os.path.join(OUT_DIR, f"oof_{args.prefix}_{args.tag}_per-fold.npy")
    test_path    = os.path.join(OUT_DIR, f"testprobs_{args.prefix}_{args.tag}.npy")

    for p in (oof_path, perfold_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    train_df = pd.read_csv(TRAIN_CSV)
    y_true   = train_df["LABEL"].values.astype(int)
    test_df  = pd.read_csv(TEST_CSV)

    oof          = np.load(oof_path)               # [N, 2]
    oof_perfold  = np.load(perfold_path)           # [F, N, 2] val-only-nonzero
    test_probs   = np.load(test_path)              # [N_test, 2]
    n_folds      = oof_perfold.shape[0]
    print(f"=== Foldwise threshold tune | tag={args.tag} ===")
    print(f"n_folds = {n_folds} | n_train = {len(y_true)} | n_test = {len(test_probs)}\n")

    # 先做全 OOF 掃描當參考
    base_pred = (oof[:, 1] >= 0.5).astype(int)
    base_f1   = f1_score(y_true, base_pred, average="macro")
    base_acc  = accuracy_score(y_true, base_pred)
    _, _, oof_best_t, oof_best_f1 = sweep_one(
        oof[:, 1], y_true, args.range[0], args.range[1], args.step
    )
    print(f"[Reference] full-OOF baseline (t=0.5): F1 = {base_f1:.4f}  acc = {base_acc:.4f}")
    print(f"[Reference] full-OOF best t = {oof_best_t:.3f}  F1 = {oof_best_f1:.4f}  Δ = {oof_best_f1 - base_f1:+.4f}")
    print()

    # 每 fold 各自掃
    fold_results = []
    for f in range(n_folds):
        # val 位置 = 該 fold 中非零的 row
        val_mask = (oof_perfold[f].sum(axis=1) > 0)
        idx = np.where(val_mask)[0]
        if len(idx) == 0:
            print(f"[fold {f}] empty val, skip")
            continue
        probs_pos = oof_perfold[f, idx, 1]
        y_va      = y_true[idx]
        f1_at_05  = f1_score(y_va, (probs_pos >= 0.5).astype(int), average="macro")
        _, _, bt, bf = sweep_one(probs_pos, y_va, args.range[0], args.range[1], args.step)
        delta = bf - f1_at_05
        fold_results.append({
            "fold": f, "n_val": len(idx),
            "f1@0.5": f1_at_05, "best_t": bt, "best_f1": bf, "delta": delta,
        })
        print(f"[fold {f}] n={len(idx):4d}  F1@0.5 = {f1_at_05:.4f}  best_t = {bt:.3f}  best_F1 = {bf:.4f}  Δ = {delta:+.4f}")

    bts = np.array([r["best_t"] for r in fold_results])
    spread = float(bts.max() - bts.min())
    median_t = float(np.median(bts))
    mean_t   = float(np.mean(bts))

    print("\n--- Convergence check ---")
    print(f"per-fold best_t: {bts.tolist()}")
    print(f"spread = {spread:.3f}   (threshold for convergence: < {CONVERGENCE_SPREAD})")
    print(f"median = {median_t:.3f}  | mean = {mean_t:.3f}")

    if args.force_t is not None:
        final_t = float(args.force_t)
        verdict = f"FORCED    → 使用 --force-t {final_t:.3f},跳過收斂判定"
    elif spread < CONVERGENCE_SPREAD:
        final_t = median_t
        verdict = f"CONVERGED → spread {spread:.3f} < {CONVERGENCE_SPREAD},採 fold-median best_t"
    else:
        final_t = 0.5
        verdict = f"DIVERGED  → spread {spread:.3f} ≥ {CONVERGENCE_SPREAD},退回 t=0.5"
    print(f"verdict: {verdict}")
    print(f"final_t = {final_t:.3f}")

    final_pred_oof = (oof[:, 1] >= final_t).astype(int)
    final_f1_oof = f1_score(y_true, final_pred_oof, average="macro")
    print(f"\nFull-OOF F1 @ final_t={final_t:.3f}: {final_f1_oof:.4f}  (Δ vs t=0.5: {final_f1_oof - base_f1:+.4f})")

    # Test side
    test_pos = test_probs[:, 1]
    test_pred_05 = (test_pos >= 0.5).astype(int)
    test_pred_ft = (test_pos >= final_t).astype(int)
    flips = int((test_pred_05 != test_pred_ft).sum())
    print(f"\nTest pred dist @ t=0.5  : neg={int((test_pred_05==0).sum())}, pos={int((test_pred_05==1).sum())}")
    print(f"Test pred dist @ t={final_t:.3f}: neg={int((test_pred_ft==0).sum())}, pos={int((test_pred_ft==1).sum())}")
    print(f"flips vs t=0.5: {flips} / {len(test_pos)} ({flips/len(test_pos)*100:.2f}%)")

    if args.write_submission:
        suffix = f"t{final_t:.3f}"
        out_path = os.path.join(OUT_DIR, f"sub_{args.prefix}_{args.tag}_{suffix}.csv")
        sub = pd.DataFrame({"row_id": test_df["row_id"].values, "LABEL": test_pred_ft})
        sub.to_csv(out_path, index=False)
        print(f"\nWrote submission: {out_path}")


if __name__ == "__main__":
    main()
