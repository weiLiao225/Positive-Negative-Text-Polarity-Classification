# 文本情緒極性分類（Positive/Negative Text Polarity Classification）

## 專案目的

本專案為 7 天 Kaggle 競賽的完整實驗紀錄，任務目標為**句子層級的二元情緒極性分類**（正面 / 負面），評估指標為 Macro-F1。

核心挑戰為訓練集（電影評論為主）與測試集（商品 / 遊戲評論為主）之間存在顯著的 **covariate shift（協變量偏移）**，導致 CV 0.871 與 Public LB 0.771 之間有約 10% 的 generalization gap。

Kaggle 競賽連結: https://www.kaggle.com/competitions/positive-negative-text-polarity-classification-26

---

## 資料集介紹

| 集合 | 筆數 | 說明 |
|---|---|---|
| `train_2022.csv` | 2,000 | 完全平衡（正 / 負各 1000）；Rotten Tomatoes 風格電影評論 + Amazon 風格商品評論各半 |
| `test_no_answer_2022.csv` | 10,999 | 無標註；以商品 / 遊戲評論為主，幾乎不含電影評論 |

- **輸入欄位**：`TEXT`（文本）
- **標籤欄位**：`LABEL`（0 = 負面，1 = 正面）
- 序列長度：train max 61 subword tokens、test max 44（以 siebert tokenizer 實測）

---

## 實驗方法與結果

### 分數演進

| 方案 | Public LB | Private LB |
|---|---|---|
| Day 1：siebert vanilla 5-fold baseline | 0.77107 | 0.77829 |
| Day 2–5：16 種訓練端修正（hard pseudo、label smoothing、FGM、stacking 等） | 全部失敗 | — |
| Day 6：soft pseudo（siebert teacher） | 0.77601 | 0.78064 |
| Day 7：soft pseudo（ensemble teacher，t=0.520） | 0.78539 | 0.80312 |
| **Day 7：soft pseudo（ensemble teacher，t=0.500）** | **0.78402** | **0.80352** |

> Private LB 揭露後 t=0.500 反超 t=0.520，最終成績 **0.80352**（+4.20% vs baseline）。
> 在kaggle上已選擇 t=0.520 版本為最終版，故 private LB 顯示 0.80312 。

### 最終方案：五階段 Soft Pseudo-Label Distillation

1. **Phase 1 — 教師集成**：以 `siebert/sentiment-roberta-large-english` 和 `google/electra-large-discriminator` 各自 5-fold 微調，對測試集產出軟機率後取算術平均作為 pseudo-label
2. **Phase 2 — 直推式資料擴增**：將全部 10,999 筆測試集（τ=0，不過濾）加入學生訓練，使模型在訓練時即接觸測試分布
3. **Phase 3 — 師生蒸餾（α=1.0）**：對真實資料用 hard CE，對 pseudo 資料用 soft CE，分段平均後加權合併
4. **Phase 4 — 嚴格 OOF 驗證**：pseudo 樣本永遠在 train side，validation 僅由 2,000 筆真實資料構成，防止 target leakage
5. **Phase 5 — 閾值調整**：per-fold best-t sweep（[0.20, 0.80]，step=0.005），以 spread < 0.15 為收斂門檻；最終以 `--force-t 0.520` 指定閾值

完整方法論與失敗分析見 [report_final.md](report_final.md)，逐日實驗紀錄見 [logs/](logs/)。

---

## 專案結構

```
├── data/                        # 原始資料（已加入 .gitignore）
│   ├── train_2022.csv
│   ├── test_no_answer_2022.csv
│   └── sample_submission.csv
├── eda/                         # 探索式資料分析腳本
│   └── EDA.py
├── eda_outputs/                 # EDA 產出圖表與統計
├── src/                         # 訓練 pipeline 與工具腳本
│   ├── pipe_siebert_vanilla.py          # siebert 5-fold baseline
│   ├── pipe_electra_vanilla.py          # ELECTRA-large 5-fold baseline
│   ├── pipe_siebert_soft_pseudo.py      # 學生模型（soft pseudo 蒸餾）
│   ├── tool_sweep_soft_pseudo.py        # alpha 掃描工具
│   └── tool_threshold_tune_foldwise.py  # per-fold 閾值調整工具
├── outputs/                     # 模型輸出（.npy 已加入 .gitignore）
│   └── sub_*.csv                # 提交檔（保留作紀錄）
├── logs/                        # 逐日實驗紀錄
├── report_final.md              # 完整技術報告
├── requirements.txt
└── .gitignore
```

---

## 如何開始

### 環境安裝

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 需要有 GPU 環境（fp16 推論）。CPU 可執行但速度極慢。

### 完整復現流程

```bash
# Step 1：訓練 siebert vanilla（教師 + baseline）
python src/pipe_siebert_vanilla.py

# Step 2：訓練 ELECTRA-large vanilla（教師）
python src/pipe_electra_vanilla.py

# Step 3：產出集成教師軟標籤
python -c "
import numpy as np
a = np.load('outputs/testprobs_siebert_vanilla.npy')
b = np.load('outputs/testprobs_electra_vanilla.npy')
np.save('outputs/testprobs_teacher-siebert-electra.npy', (a + b) / 2)
"

# Step 4：訓練學生模型（soft pseudo，集成教師）
python src/pipe_siebert_soft_pseudo.py --tau 0.0 --alpha 1.0 \
  --teacher outputs/testprobs_teacher-siebert-electra.npy \
  --tag teacher-ens

# Step 5：閾值調整並輸出提交檔
python src/tool_threshold_tune_foldwise.py --tag teacher-ens \
  --force-t 0.520 --write-submission
# → outputs/sub_siebert_soft-pseudo_teacher-ens_t0.520.csv
```

所有 pipeline 固定使用 `seed=42, n_splits=5, shuffle=True` Stratified 5-Fold，確保各模型的 fold 切分完全一致，維持 OOF 與 teacher prediction 的資料對齊。
