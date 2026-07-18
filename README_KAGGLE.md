# Kaggle Setup Guide

This guide explains how to run the complaint span extraction experiments on Kaggle Notebooks.

---

## Prerequisites

- Kaggle account with GPU access enabled
- Internet enabled in notebook settings

---

## Step 1: Upload Repository to GitHub

On your local machine:

```bash
cd paper_thangvu

# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit: complaint span extraction project"

# Create repo on GitHub, then:
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git branch -M main
git push -u origin main
```

---

## Step 2: Open Kaggle Notebook

1. Go to [kaggle.com](https://kaggle.com)
2. Click **+ New Notebook**
3. In the notebook menu: **File → Add utility script** or clone directly

---

## Step 3: Enable GPU and Internet

1. In the notebook toolbar, click **Settings** (gear icon)
2. Select **GPU T4** (or similar) as the accelerator
3. Ensure **Internet** is turned ON
4. Set **Persistence** to "Variables and Files" for faster re-runs

---

## Step 4: Clone the Repository

In a Kaggle code cell:

```bash
!git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
cd <YOUR_REPO>
```

This will clone the entire repository including data, scripts, and configs.

---

## Step 5: Install Dependencies

```bash
cd <YOUR_REPO>
!pip install -r requirements.txt -q
```

Expected time: 3-5 minutes.

---

## Step 6: Verify Data Integrity

```bash
!python scripts/check_data_integrity.py
```

Expected output: All checks PASSED.

---

## Step 7: Run Direct Baselines

```bash
!bash scripts/run_direct_baselines.sh
```

This trains 4 models:
- `phobert_ce` — PhoBERT-base-v2 with cross-entropy loss
- `phobert_weighted_ce` — PhoBERT-base-v2 with weighted cross-entropy
- `xlm_roberta_ce` — XLM-RoBERTa-base with cross-entropy
- `mbert_ce` — mBERT-base-cased with cross-entropy

Expected training time: ~15-30 minutes per model (depending on GPU).

Results will be saved to `outputs/experiments/direct_baselines/`.

---

## Step 8: Run Transfer Learning

```bash
!bash scripts/run_transfer_learning.sh
```

This runs 3 transfer strategies:
- `aux_all_then_complaint` — Pretrain on all auxiliary data, fine-tune on complaint spans
- `causasent_then_complaint` — Pretrain on CausaSent-ATE-v2 only
- `uvisd4sa_then_complaint` — Pretrain on UIT-ViSD4SA only

Expected training time: ~20-40 minutes per strategy.

Results will be saved to `outputs/experiments/transfer_learning/`.

---

## OOM (Out of Memory) Fallback

If you encounter CUDA OOM errors, reduce resource usage in this order:

### Option 1: Reduce batch size
```yaml
# In configs/direct_baselines.yaml or configs/transfer_learning.yaml
train_batch_size: 16  # try 8, then 4, then 2
```

### Option 2: Reduce max length
```yaml
max_length: 256  # try 192, then 128
```

### Option 3: Disable FP16
```yaml
fp16: false
```

### Option 4: Reduce epochs
```yaml
num_train_epochs: 3  # minimum viable
```

### Option 5: Use smaller model
```yaml
# Replace vinai/phobert-base-v2 with vinai/phobert-base
# Or use xlm-roberta-base with reduced max_length
```

---

## Storage-Saving Mode

This repository does **not save model checkpoints or weights** by default. Only metrics, predictions, and error analysis are written to disk.

**Why?** Kaggle has a 20 GB output limit. Saving full model checkpoints (~1-2 GB each × 4 models × 3 strategies) would exceed this limit quickly.

**What is saved** (per experiment):
- `test_metrics.json` — entity-level and token-level metrics
- `test_predictions.jsonl` — per-record predictions and ground truth
- `error_analysis.csv` — error analysis by record
- `per_label_report.csv` — per-label precision/recall/F1
- `train_log.txt` — training configuration and timing
- `completed_result.json` — marker file confirming completion
- `experiment_results.csv` — summary table across all models

**What is NOT saved:**
- No `checkpoint-*` folders
- No `pytorch_model.bin` or `.safetensors` files
- No HuggingFace model checkpoints
- No optimizer/scheduler states

**In-memory transfer learning:** The transfer learning pipeline keeps encoder weights in GPU/CPU memory between auxiliary pretraining and complaint fine-tuning. The auxiliary model checkpoint is never written to disk.

**If you want to save model weights later**, edit `configs/direct_baselines.yaml` or `configs/transfer_learning.yaml`:

```yaml
save_strategy: "epoch"
save_total_limit: 1
save_model: true
load_best_model_at_end: true
metric_for_best_model: entity_f1
```

---

## Expected Output Structure

```
outputs/experiments/
├── direct_baselines/
│   ├── phobert_ce/
│   │   ├── completed_result.json
│   │   ├── test_metrics.json
│   │   ├── test_predictions.jsonl
│   │   ├── error_analysis.csv
│   │   ├── per_label_report.csv
│   │   └── train_log.txt
│   ├── phobert_weighted_ce/
│   ├── xlm_roberta_ce/
│   ├── mbert_ce/
│   └── experiment_results.csv
│
└── transfer_learning/
    ├── aux_all_then_complaint/
    │   ├── test_metrics.json
    │   ├── test_predictions.jsonl
    │   ├── error_analysis.csv
    │   ├── train_log.txt
    │   └── completed_result.json
    ├── causasent_then_complaint/
    └── uvisd4sa_then_complaint/
```

**Estimated output size:** ~5-20 MB per experiment (metrics + predictions only, no model weights).

---

## Key Metrics

- **entity_f1** — Primary metric. Strict span-level F1 (exact start+end match)
- **entity_precision** — Span-level precision
- **entity_recall** — Span-level recall
- **token_f1_macro** — Token-level macro F1 (seqeval)
- **token_accuracy** — Token-level accuracy

---

## Downloading Results

After training, download results from Kaggle:

1. Click the **Data** tab in the right panel
2. Navigate to `outputs/experiments/`
3. Download individual files or the entire folder

Or use the Kaggle API in a cell:

```bash
!zip -r experiment_results.zip outputs/
```

---

## Troubleshooting

### `CUDA out of memory`
→ Reduce `train_batch_size` to 8, 4, or 2 in the config.

### `Internet is off`
→ Enable Internet in Notebook Settings → Internet.

### `Module not found: transformers`
→ Re-run `!pip install -r requirements.txt`.

### Model download fails
→ Kaggle sometimes blocks large model downloads. Try:
```bash
!pip install --no-cache-dir transformers datasets accelerate
```

### Slow training
→ Ensure GPU is enabled. Check with:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Smoke Test (Fast Validation)

If you want to verify the pipeline works before running full experiments:

```bash
!python scripts/train_direct_baselines.py \
    --config configs/direct_baselines.yaml \
    --models phobert_ce \
    --smoke_test
```

This runs with 32 records, 1 epoch — completes in ~2-3 minutes.
