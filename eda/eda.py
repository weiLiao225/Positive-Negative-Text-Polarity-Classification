import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 視覺化風格設定
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# 1. 定義路徑 (使用專案本地檔案)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
TRAIN_PATH = os.path.join(PROJECT_DIR, "data", "train_2022.csv")
TEST_PATH = os.path.join(PROJECT_DIR, "data", "test_no_answer_2022.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "eda_outputs")

LABEL_COL = "LABEL"
TEXT_COL = "TEXT"
SIEBERT_MODEL_ID = "siebert/sentiment-roberta-large-english"


def _ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案：{path}")
    return pd.read_csv(path)


def _annotate_countplot(ax) -> None:
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main(show_plots: bool = False) -> None:
    _ensure_output_dir(OUTPUT_DIR)

    train_df = _load_csv(TRAIN_PATH)
    test_df = _load_csv(TEST_PATH)

    print(f"✅ 訓練集筆數: {len(train_df)}")
    print(f"✅ 測試集筆數: {len(test_df)}")

    # 計算字元長度
    train_df["text_length"] = train_df[TEXT_COL].astype(str).apply(len)
    test_df["text_length"] = test_df[TEXT_COL].astype(str).apply(len)

    # ---------------------------------------------------------
    # 分析 1：正反面標籤的數量分佈
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.countplot(
        data=train_df,
        x=LABEL_COL,
        hue=LABEL_COL,
        ax=axes[0],
        palette="Set2",
        legend=False,
    )
    axes[0].set_title("Polarity Class Distribution (LABEL)", fontsize=14)
    axes[0].set_xlabel("Polarity", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    _annotate_countplot(axes[0])

    # ---------------------------------------------------------
    # 分析 2：句子長度分佈 (依標籤)
    # ---------------------------------------------------------
    sns.histplot(
        data=train_df,
        x="text_length",
        hue=LABEL_COL,
        bins=50,
        kde=True,
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("Text Length Distribution by Polarity", fontsize=14)
    axes[1].set_xlabel("Text Length (Characters)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "train_label_and_length.png")
    fig.savefig(fig_path, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)

    # ---------------------------------------------------------
    # 視覺化：訓練集與測試集長度分佈對比
    # ---------------------------------------------------------
    combined_df = pd.concat(
        [
            train_df[["text_length"]].assign(Dataset="Train"),
            test_df[["text_length"]].assign(Dataset="Test"),
        ],
        ignore_index=True,
    )

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=combined_df,
        x="text_length",
        hue="Dataset",
        bins=50,
        kde=True,
        element="step",
        palette="husl",
        ax=ax2,
    )
    ax2.set_title("Text Length Distribution: Train vs Test", fontsize=14)
    ax2.set_xlabel("Text Length (Characters)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)

    fig2.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "train_vs_test_length.png")
    fig2.savefig(fig2_path, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig2)

    # ---------------------------------------------------------
    # 數值分析：百分位數統計
    # ---------------------------------------------------------
    stats = pd.DataFrame(
        {
            "Train": train_df["text_length"].describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            ),
            "Test": test_df["text_length"].describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            ),
        }
    )
    print("\n📊 文本長度詳細統計 (百分位數):")
    print(stats)

    stats_path = os.path.join(OUTPUT_DIR, "length_stats.csv")
    stats.to_csv(stats_path, index=True)

    # ---------------------------------------------------------
    # Token 數量分析 (SieBERT tokenizer)
    # ---------------------------------------------------------
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "缺少 transformers 套件，請先安裝：pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(SIEBERT_MODEL_ID, use_fast=True)

    def _token_lengths(texts: pd.Series) -> pd.Series:
        enc = tokenizer(
            texts.astype(str).tolist(),
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        return pd.Series([len(ids) for ids in enc["input_ids"]])

    train_token_len = _token_lengths(train_df[TEXT_COL])
    test_token_len = _token_lengths(test_df[TEXT_COL])

    token_stats = pd.DataFrame(
        {
            "Train": {
                "max": int(train_token_len.max()),
                "p99": float(train_token_len.quantile(0.99)),
                "p95": float(train_token_len.quantile(0.95)),
                "median": float(train_token_len.median()),
            },
            "Test": {
                "max": int(test_token_len.max()),
                "p99": float(test_token_len.quantile(0.99)),
                "p95": float(test_token_len.quantile(0.95)),
                "median": float(test_token_len.median()),
            },
        }
    )

    print("\n📊 Token 數量統計 (SieBERT tokenizer):")
    print(token_stats)

    truncation_candidates = [32, 64, 96, 128]
    trunc_table = pd.DataFrame(
        {
            "max_length": truncation_candidates,
            "train_trunc_rate": [
                float((train_token_len > m).mean()) for m in truncation_candidates
            ],
            "test_trunc_rate": [
                float((test_token_len > m).mean()) for m in truncation_candidates
            ],
        }
    )
    print("\n📊 Token 截斷率 (SieBERT tokenizer):")
    print(trunc_table)

    token_plot_df = pd.concat(
        [
            train_token_len.to_frame(name="token_length").assign(Dataset="Train"),
            test_token_len.to_frame(name="token_length").assign(Dataset="Test"),
        ],
        ignore_index=True,
    )

    max_token_len = int(max(train_token_len.max(), test_token_len.max()))

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=token_plot_df,
        x="token_length",
        hue="Dataset",
        binwidth=1,
        binrange=(1, max_token_len + 1),
        discrete=True,
        kde=False,
        element="bars",
        alpha=0.4,
        palette="husl",
        ax=ax3,
    )
    ax3.set_title("Token Length Distribution: Train vs Test (SieBERT)", fontsize=14)
    ax3.set_xlabel("Token Length", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)

    fig3.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "train_vs_test_token_length.png")
    fig3.savefig(fig3_path, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig3)
    print(f"\n✅ 圖片與統計已輸出至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main(show_plots=False)