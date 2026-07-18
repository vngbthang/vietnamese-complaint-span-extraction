# From Complaint Detection to Complaint Span Extraction in Vietnamese Customer Reviews

## Project Overview

This repository contains code and data for **complaint span extraction** from Vietnamese customer reviews using transformer-based models.

### Task Definition

Given a Vietnamese customer review, identify all text spans that describe complaints. Each complaint span is annotated using BIO format:

| Label | Description |
|-------|-------------|
| `O` | Outside any complaint span |
| `B-COMP` | Beginning of a complaint span |
| `I-COMP` | Inside (continuation of) a complaint span |

Example:

```
Review: "Máy pin trâu nhưng camera kém và loa bị rè"
Tokens: ["Máy", "pin", "trâu", "nhưng", "camera", "kém", "và", "loa", "bị", "rè"]
BIO:    ["O", "O", "O", "O", "B-COMP", "I-COMP", "O", "B-COMP", "I-COMP", "I-COMP"]
```

### Datasets

#### Main Target Dataset: ViOCD-Span

A cleaned BIO-formatted dataset for **complaint span extraction** in Vietnamese.
Located at `data/complaint_span_bio_clean/`.

```
train.jsonl   : 4,387 records
valid.jsonl   :   548 records
test.jsonl    :   549 records
────────────────────────────────
Total         : 5,484 records
Total spans   : 5,237
Labels        : O, B-COMP, I-COMP
```

#### Auxiliary Datasets: UIT-ViSD4SA + CausaSent-ATE-v2

Pre-training datasets for **aspect term extraction**, used for transfer learning.
Located at `data/auxiliary_ate_bio_clean/`.

| Source | Records | Aspect Spans |
|--------|---------|-------------|
| UIT-ViSD4SA | 10,892 | 33,625 |
| CausaSent-ATE-v2 | 7,066 | 9,989 |
| **Total** | **17,958** | **43,614** |

**Important:** Aspect spans ≠ Complaint spans. The auxiliary datasets contain general aspect terms (e.g., "camera", "pin", "loa"), not specifically complaints. They are used solely for **pre-training** the encoder before fine-tuning on complaint span extraction.

Labels for auxiliary data: `O`, `B-ASP`, `I-ASP`

### Repository Structure

```
.
├── configs/                          # Training configs
│   ├── direct_baselines.yaml
│   └── transfer_learning.yaml
├── data/
│   ├── complaint_span_bio_clean/    # Main target dataset (O, B-COMP, I-COMP)
│   └── auxiliary_ate_bio_clean/     # Auxiliary dataset (O, B-ASP, I-ASP)
├── scripts/
│   ├── check_data_integrity.py       # Verify data quality
│   ├── train_direct_baselines.py     # Train without transfer learning
│   ├── train_transfer_learning.py    # Train with auxiliary pretraining
│   ├── run_direct_baselines.sh
│   └── run_transfer_learning.sh
├── src/
│   ├── __init__.py
│   ├── data_utils.py                # Data loading & BioRecord class
│   ├── tokenization_utils.py         # Subword alignment
│   ├── metrics.py                    # Span & token metrics
│   ├── models.py                     # Model definitions
│   ├── trainer_utils.py              # Training utilities
│   └── error_analysis.py             # Error analysis
├── outputs/                          # Experiment outputs
├── requirements.txt
├── README.md
└── README_KAGGLE.md
```

### Models

| Model | Language | Use Case |
|-------|----------|---------|
| `vinai/phobert-base-v2` | Vietnamese | Primary model |
| `xlm-roberta-base` | Multilingual | Cross-lingual baseline |
| `bert-base-multilingual-cased` | Multilingual | Baseline |

### Experiments

#### Direct Baselines (Step 6)
Train models directly on ViOCD-Span without auxiliary data:
- PhoBERT-base-v2 + Cross-Entropy
- PhoBERT-base-v2 + Weighted Cross-Entropy
- XLM-RoBERTa-base + Cross-Entropy
- mBERT-base-cased + Cross-Entropy

#### Transfer Learning (Step 7)
Pre-train encoder on auxiliary aspect data, then fine-tune on complaint spans:
- `aux_all_then_complaint` — All auxiliary data → Complaint data
- `causasent_then_complaint` — CausaSent-ATE-v2 only → Complaint data
- `uvisd4sa_then_complaint` — UIT-ViSD4SA only → Complaint data

### Key Design Decisions

1. **BIO format** for span representation (standard for token classification).
2. **Strict entity-level metrics** — spans must match exactly at both start and end token positions.
3. **Pre-tokenized whitespace tokens** — the dataset provides pre-tokenized tokens and BIO labels aligned at the whitespace-token level. Subword alignment is handled by HuggingFace `is_split_into_words=True`.
4. **Separate label spaces** — auxiliary data uses `B-ASP/I-ASP`, complaint data uses `B-COMP/I-COMP`. Only encoder weights are transferred; classification heads are reinitialized.
5. **Aspect ≠ Complaint** — auxiliary datasets are general aspect term datasets, not complaint-specific. They provide linguistic pre-training signal only.
6. **No model checkpoints saved** — only metrics, predictions, and error analysis are written to disk. This keeps output sizes small (~few MB per experiment) and avoids Kaggle storage issues. Encoder weights are transferred in-memory between auxiliary pretraining and complaint fine-tuning.

### Citation

If you use this dataset or code, please cite:

```
# TODO: Add citation when paper is published
```

### License

TODO: Add license information.
