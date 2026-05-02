"""
Soft-pseudo alpha sweep driver
==============================

固定 tau,掃多個 alpha,自動跑 pipe_siebert_soft-pseudo.py,
最後彙總 OOF F1 / per-fold best_t spread / 推薦配置。

設計理念:
  - tau=0 已確認是「最軟」的設定(Day 6 OOF 0.8890 訊號)
  - 沿 alpha 軸掃 {0.3, 0.5, 1.0, 1.5, 2.0} 找 OOF F1 高峰
  - 跳過已存在的 tag(可中斷再續跑)
  - 結尾印推薦表

用法:
  python tool_sweep-soft-pseudo.py
  python tool_sweep-soft-pseudo.py --alphas 0.3 0.5 1.0 1.5 2.0 --tau 0.0
  python tool_sweep-soft-pseudo.py --alphas 0.5 1.0 --dry-run    # 只列要跑的 tag
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TRAIN_CSV    = "data/train_2022.csv"
OUT_DIR   = "outputs"
PIPE      = "pipe_siebert_soft-pseudo.py"
PREFIX    = "siebert_soft-pseudo"


def tag_for(tau: float, alpha: float) -> str:
    return f"tau{tau:.2f}_a{alpha:.2f}".replace(".", "")


def files_for(tag: str):
    return (
        f"{OUT_DIR}/oof_{PREFIX}_{tag}.npy",
        f"{OUT_DIR}/oof_{PREFIX}_{tag}_per-fold.npy",
        f"{OUT_DIR}/testprobs_{PREFIX}_{tag}.npy",
    )


def perfold_best_t(oof_perfold, y_true, lo=0.20, hi=0.80, step=0.005):
    bts, bfs = [], []
    n_folds = oof_perfold.shape[0]
    ts = np.arange(lo, hi + 1e-9, step)
    for f in range(n_folds):
        mask = oof_perfold[f].sum(axis=1) > 0
        idx = np.where(mask)[0]
        probs_pos = oof_perfold[f, idx, 1]
        y_va = y_true[idx]
        f1s = [f1_score(y_va, (probs_pos >= t).astype(int), average="macro") for t in ts]
        bi = int(np.argmax(f1s))
        bts.append(float(ts[bi]))
        bfs.append(float(f1s[bi]))
    return bts, bfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.3, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--dry-run", action="store_true",
                    help="只列要跑的 tag,不真的訓練")
    ap.add_argument("--skip-train", action="store_true",
                    help="不訓練,只彙總已存在的 oof 檔")
    args = ap.parse_args()

    print(f"=== Sweep | tau={args.tau} | alphas={args.alphas} ===\n")

    train_df = pd.read_csv(TRAIN_CSV)
    y_true = train_df["LABEL"].values.astype(int)

    plan = []
    for alpha in args.alphas:
        tag = tag_for(args.tau, alpha)
        oof_p, perfold_p, test_p = files_for(tag)
        exists = all(os.path.exists(p) for p in (oof_p, perfold_p, test_p))
        plan.append((alpha, tag, exists))

    print("Plan:")
    for alpha, tag, exists in plan:
        marker = "[CACHED]" if exists else "[TRAIN ]"
        print(f"  {marker} alpha={alpha:>4.2f}  tag={tag}")
    print()

    if args.dry_run:
        return

    # 跑訓練
    if not args.skip_train:
        for alpha, tag, exists in plan:
            if exists:
                print(f"Skip {tag}: cached")
                continue
            print(f"\n>>> Training alpha={alpha} (tag={tag}) ...")
            cmd = [sys.executable, PIPE,
                   "--tau", str(args.tau), "--alpha", str(alpha)]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"!!! pipe failed for alpha={alpha}, abort sweep")
                sys.exit(ret.returncode)

    # 彙總
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    rows = []
    for alpha, tag, _ in plan:
        oof_p, perfold_p, _ = files_for(tag)
        if not os.path.exists(oof_p):
            print(f"[skip] {tag}: oof file missing")
            continue
        oof = np.load(oof_p)
        oof_perfold = np.load(perfold_p)

        f1_05 = f1_score(y_true, (oof[:, 1] >= 0.5).astype(int), average="macro")
        ts = np.arange(0.20, 0.80 + 1e-9, 0.005)
        f1s_full = [f1_score(y_true, (oof[:, 1] >= t).astype(int), average="macro") for t in ts]
        bi = int(np.argmax(f1s_full))
        f1_oof_best_t = float(ts[bi])
        f1_oof_best   = float(f1s_full[bi])

        bts, bfs = perfold_best_t(oof_perfold, y_true)
        spread = max(bts) - min(bts)
        median_t = float(np.median(bts))

        rows.append({
            "alpha": alpha,
            "F1@0.5": f1_05,
            "OOF_best_t": f1_oof_best_t,
            "OOF_best_F1": f1_oof_best,
            "fold_med_t": median_t,
            "fold_spread": spread,
            "fold_best_t": str([round(b, 3) for b in bts]),
        })

    if not rows:
        print("No results to summarize.")
        return

    df = pd.DataFrame(rows).sort_values("F1@0.5", ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n--- Recommendation ---")
    best_row = df.iloc[0]
    print(f"Best by argmax F1: alpha={best_row['alpha']}, OOF F1@0.5 = {best_row['F1@0.5']:.4f}")
    converged = df[df["fold_spread"] < 0.15]
    if not converged.empty:
        cb = converged.sort_values("OOF_best_F1", ascending=False).iloc[0]
        print(f"Best converged (spread<0.15): alpha={cb['alpha']}, "
              f"OOF F1 @ t={cb['fold_med_t']:.3f} = {cb['OOF_best_F1']:.4f}")
    else:
        print("No alpha gave fold-convergent threshold (spread<0.15); use t=0.5 for all.")


if __name__ == "__main__":
    main()
