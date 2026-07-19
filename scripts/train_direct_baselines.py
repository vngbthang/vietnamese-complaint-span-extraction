#!/usr/bin/env python3
"""
Direct baselines training for complaint span extraction.

NO CHECKPOINTS ARE SAVED.
Only metrics, predictions, and reports are written to disk.

Trains token classification models on:
data/complaint_span_bio_clean/
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import glob
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_utils import load_split, safe_get
from src.metrics import (
    bio_to_spans,
    save_error_analysis,
    save_per_label_report,
    save_predictions_jsonl,
    seqeval_f1,
    seqeval_precision,
    seqeval_recall,
    seqeval_report,
)
from src.tokenization_utils import encode_tokens_with_labels, encode_tokens_with_labels_sanity_check
from src.trainer_utils import make_training_args, normalize_training_config

LABEL_SET = ['O', 'B-COMP', 'I-COMP']


# ---------------------------------------------------------------
# Cleanup utilities
# ---------------------------------------------------------------

def cleanup_checkpoints(output_dir: str) -> Dict[str, int]:
    """
    Remove all checkpoint folders and model weight files from output_dir.
    Returns counts of removed items.
    """
    removed = {'checkpoint_folders': 0, 'weight_files': 0}

    if not os.path.exists(output_dir):
        return removed

    # Remove checkpoint-* folders
    for pattern in [
        os.path.join(output_dir, 'checkpoint-*'),
        os.path.join(output_dir, '**', 'checkpoint-*'),
    ]:
        for ckpt_dir in glob.glob(pattern, recursive=True):
            if os.path.isdir(ckpt_dir):
                os.system(f'rm -rf {repr(ckpt_dir)}')
                removed['checkpoint_folders'] += 1

    # Remove weight files
    weight_patterns = [
        '*.safetensors', '*.bin', '*.pt', '*.pth',
        'pytorch_model.bin', 'model.safetensors',
        'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
        'training_args.bin', 'trainer_state.json',
        'scheduler_state.json',
    ]
    for pattern in weight_patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)
                removed['weight_files'] += 1
        for path in glob.glob(os.path.join(output_dir, '**', pattern), recursive=True):
            if os.path.isfile(path):
                os.remove(path)
                removed['weight_files'] += 1

    return removed


def get_output_size(output_dir: str) -> Dict[str, Any]:
    """Get size of output directory and count checkpoint/model files."""
    total_size = 0
    checkpoint_folders = 0
    weight_files = 0
    all_files = 0

    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            # Count checkpoint folders
            checkpoint_folders += sum(1 for d in dirs if d.startswith('checkpoint-'))
            for f in files:
                all_files += 1
                fpath = os.path.join(root, f)
                try:
                    total_size += os.path.getsize(fpath)
                except OSError:
                    pass
                # Count weight files
                if any(f.endswith(ext) for ext in [
                    '.safetensors', '.bin', '.pt', '.pth', '.ckpt',
                    'pytorch_model.bin', 'model.safetensors',
                ]):
                    weight_files += 1

    return {
        'total_bytes': total_size,
        'total_mb': total_size / 1e6,
        'checkpoint_folders': checkpoint_folders,
        'weight_files': weight_files,
        'all_files': all_files,
    }


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train direct baseline models (no checkpoints)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--models', type=str, nargs='+',
                        choices=['phobert_ce', 'phobert_weighted_ce',
                                 'xlm_roberta_ce', 'mbert_ce'],
                        default=None)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--force_rerun', action='store_true')
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------

def load_dataset(data_dir: str, split: str):
    return load_split(data_dir, split)


def prepare_hf_dataset(records, tokenizer, label2id, max_length: int = 256):
    """Convert BioRecords to HuggingFace Dataset.

    Uses encode_tokens_with_labels() which works for both fast and slow
    tokenizers, avoiding direct .word_ids() calls.
    """
    encoded_records = []
    for rec in records:
        enc = encode_tokens_with_labels(
            tokenizer,
            rec.tokens,
            rec.bio_tags,
            label2id,
            max_length,
        )
        encoded_records.append(enc)

    dataset_dict = {
        "input_ids": [r["input_ids"] for r in encoded_records],
        "attention_mask": [r["attention_mask"] for r in encoded_records],
        "labels": [r["labels"] for r in encoded_records],
    }
    return Dataset.from_dict(dataset_dict)


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def compute_metrics_fn(id2label):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_tags, pred_tags = [], []
        for seq_pred, seq_labels in zip(predictions, labels):
            t_seq, p_seq = [], []
            for p, l in zip(seq_pred, seq_labels):
                if l != -100:
                    t_seq.append(id2label.get(l, 'O'))
                    p_seq.append(id2label.get(p, 'O'))
            true_tags.append(t_seq)
            pred_tags.append(p_seq)

        token_p = seqeval_precision(true_tags, pred_tags)
        token_r = seqeval_recall(true_tags, pred_tags)
        token_f1 = seqeval_f1(true_tags, pred_tags)

        all_gt, all_pred = [], []
        for gt, pred in zip(true_tags, pred_tags):
            all_gt.extend(bio_to_spans(gt))
            all_pred.extend(bio_to_spans(pred))

        pred_set = set(all_pred)
        gt_set = set(all_gt)
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        span_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        span_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        span_f1 = (2 * span_p * span_r / (span_p + span_r)
                     if (span_p + span_r) > 0 else 0.0)

        # Token-level accuracy
        flat_true = [t for seq in true_tags for t in seq]
        flat_pred = [p for seq in pred_tags for p in seq]
        token_acc = sum(1 for a, b in zip(flat_true, flat_pred) if a == b) / len(flat_true) if flat_true else 0.0

        return {
            'entity_precision': span_p,
            'entity_recall': span_r,
            'entity_f1': span_f1,
            'token_precision_macro': token_p,
            'token_recall_macro': token_r,
            'token_f1_macro': token_f1,
            'token_accuracy': token_acc,
        }
    return compute_metrics


# ---------------------------------------------------------------
# Training (no checkpoints)
# ---------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, label2id: Dict, id2label: Dict, device: torch.device):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(label2id)
    config.label2id = label2id
    config.id2label = id2label
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True)
    model.to(device)
    return model


def train_model(
    model_key: str,
    model_config: Dict,
    train_dataset,
    valid_dataset,
    test_dataset,
    test_records: List,
    output_dir: str,
    config: Dict,
    device: torch.device,
    label2id: Dict,
    id2label: Dict,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> bool:
    """Train a single model without saving checkpoints."""
    model_name = model_config['model_name']
    loss_type = model_config.get('loss', 'ce')
    run_dir = os.path.join(output_dir, model_key)
    os.makedirs(run_dir, exist_ok=True)

    if not force_rerun and os.path.exists(os.path.join(run_dir, 'completed_result.json')):
        print(f"  [SKIP] {model_key} already trained.")
        return True

    print(f"\n  Training {model_key} ({model_name}, loss={loss_type})...")
    print(f"  Output: {run_dir}")
    print(f"  NOTE: No checkpoints will be saved.")

    epochs = 1 if smoke_test else int(config.get('num_train_epochs', 5))

    training_args = make_training_args(
        output_dir=run_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=(
            4 if smoke_test else int(config.get('train_batch_size', 16))),
        per_device_eval_batch_size=(
            8 if smoke_test else int(config.get('eval_batch_size', 16))),
        gradient_accumulation_steps=int(config.get('gradient_accumulation_steps', 1)),
        learning_rate=float(config.get('learning_rate', 2e-5)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        warmup_ratio=float(config.get('warmup_ratio', 0.06)),
        fp16=config.get('fp16', True) and device.type == 'cuda',
        # ---- NO CHECKPOINT SAVING ----
        save_strategy='no',
        save_total_limit=0,
        load_best_model_at_end=False,
        # -------------------------------
        logging_dir=os.path.join(run_dir, 'logs'),
        logging_steps=config.get('logging_steps', 50),
        eval_strategy='epoch',
        report_to='none',
        seed=int(config.get('seed', 42)),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        run_name=model_key,
        disable_tqdm=False,
    )

    model = build_model(model_name, label2id, id2label, device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics_fn(id2label),
        data_collator=DataCollatorForTokenClassification(
            AutoTokenizer.from_pretrained(model_name),
            pad_to_multiple_of=8 if training_args.fp16 else None,
        ),
    )

    start_time = time.time()
    try:
        trainer.train()
    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        trainer.model = None
        torch.cuda.empty_cache()
        raise

    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.1f}s")

    # Save train log
    with open(os.path.join(run_dir, 'train_log.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_key}\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Loss: {loss_type}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Train time: {train_time:.1f}s\n")
        f.write(f"NO_CHECKPOINTS_SAVED=True\n")
        f.write(f"Config: {json.dumps(model_config)}\n")

    # Evaluate on test
    print(f"  Evaluating on test set...")
    test_output = trainer.predict(test_dataset)
    preds = np.argmax(test_output.predictions, axis=2)

    # Convert to tags
    true_tags_all, pred_tags_all = [], []
    for seq_pred, seq_labels in zip(preds, test_output.label_ids):
        t_seq, p_seq = [], []
        for p, l in zip(seq_pred, seq_labels):
            if l != -100:
                t_seq.append(id2label.get(l, 'O'))
                p_seq.append(id2label.get(p, 'O'))
        true_tags_all.append(t_seq)
        pred_tags_all.append(p_seq)

    # Token-level
    token_p = seqeval_precision(true_tags_all, pred_tags_all)
    token_r = seqeval_recall(true_tags_all, pred_tags_all)
    token_f1 = seqeval_f1(true_tags_all, pred_tags_all)
    flat_true = [t for seq in true_tags_all for t in seq]
    flat_pred = [p for seq in pred_tags_all for p in seq]
    token_acc = sum(1 for a, b in zip(flat_true, flat_pred) if a == b) / len(flat_true) if flat_true else 0.0

    # Weighted token F1
    token_f1_w = seqeval_f1(true_tags_all, pred_tags_all, average='weighted')

    # Span-level
    all_gt_spans, all_pred_spans = [], []
    for gt, pred in zip(true_tags_all, pred_tags_all):
        all_gt_spans.extend(bio_to_spans(gt))
        all_pred_spans.extend(bio_to_spans(pred))

    pred_set = set(all_pred_spans)
    gt_set = set(all_gt_spans)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    span_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    span_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    span_f1 = (2 * span_p * span_r / (span_p + span_r)
                if (span_p + span_r) > 0 else 0.0)

    test_metrics = {
        'model_key': model_key,
        'model_name': model_name,
        'loss': loss_type,
        'num_train_epochs': epochs,
        'entity_precision': span_p,
        'entity_recall': span_r,
        'entity_f1': span_f1,
        'entity_tp': tp,
        'entity_fp': fp,
        'entity_fn': fn,
        'token_precision_macro': token_p,
        'token_recall_macro': token_r,
        'token_f1_macro': token_f1,
        'token_f1_weighted': token_f1_w,
        'token_accuracy': token_acc,
        'train_time_seconds': train_time,
    }

    # Save metrics FIRST so they survive even if post-processing fails.
    test_metrics_path = os.path.join(run_dir, 'test_metrics.json')
    try:
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
    except Exception as e:
        print(f"  [WARN] Failed to save test_metrics.json: {e}")

    # Post-processing: predictions and error analysis.
    # Wrapped so that a failure here does not discard the metrics above.
    post_processing_ok = True
    try:
        pred_path = os.path.join(run_dir, 'test_predictions.jsonl')
        save_predictions_jsonl(test_records, pred_tags_all, true_tags_all, pred_path)
        test_metrics['prediction_path'] = pred_path
    except Exception as e:
        post_processing_ok = False
        print(f"  [WARN] Failed to save test_predictions.jsonl: {e}")

    try:
        err_path = os.path.join(run_dir, 'error_analysis.csv')
        save_error_analysis(test_records, pred_tags_all, true_tags_all, err_path)
        test_metrics['error_analysis_path'] = err_path
    except Exception as e:
        post_processing_ok = False
        print(f"  [WARN] Failed to save error_analysis.csv: {e}")

    # Per-label report
    try:
        report = seqeval_report(true_tags_all, pred_tags_all, digits=4, output_dict=True)
        per_label_path = os.path.join(run_dir, 'per_label_report.csv')
        save_per_label_report({
            'per_label': {
                k: {'precision': v.get('precision', 0), 'recall': v.get('recall', 0),
                    'f1': v.get('f1', 0), 'support': v.get('support', 0)}
                for k, v in report.items() if isinstance(v, dict)
            }
        }, per_label_path)
        test_metrics['per_label_report_path'] = per_label_path
    except Exception as e:
        post_processing_ok = False
        print(f"  [WARN] Failed to save per_label_report.csv: {e}")

    # Re-save metrics with post-processing paths appended (if any succeeded).
    try:
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
    except Exception:
        pass

    # Only mark completed if all post-processing succeeded.
    if post_processing_ok:
        try:
            with open(os.path.join(run_dir, 'completed_result.json'), 'w') as f:
                json.dump({'model_key': model_key, 'model_name': model_name,
                           'test_metrics': test_metrics, 'train_time': train_time}, f, indent=2)
        except Exception as e:
            print(f"  [WARN] Failed to save completed_result.json: {e}")
    else:
        print("  [WARN] Post-processing had errors; completed_result.json not written.")

    # ---- Cleanup: delete model from trainer, free memory ----
    del trainer
    del model
    torch.cuda.empty_cache()

    # Cleanup checkpoints if any were accidentally created
    cleanup = config.get('cleanup_checkpoints', True)
    if cleanup:
        removed = cleanup_checkpoints(run_dir)
        if removed['checkpoint_folders'] > 0 or removed['weight_files'] > 0:
            print(f"  [CLEANUP] Removed {removed['checkpoint_folders']} checkpoint folders, "
                  f"{removed['weight_files']} weight files")

    # Print size info
    size_info = get_output_size(run_dir)
    print(f"  Output size: {size_info['total_mb']:.2f} MB "
          f"({size_info['all_files']} files, "
          f"{size_info['checkpoint_folders']} checkpoints, "
          f"{size_info['weight_files']} weight files)")

    print(f"  Results saved to {run_dir}")
    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)
    config = normalize_training_config(config)

    output_dir = config.get('output_dir', 'outputs/experiments/direct_baselines')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(int(config.get('seed', 42)))

    print("Config type check:")
    for k in ["learning_rate", "weight_decay", "warmup_ratio", "num_train_epochs",
               "train_batch_size", "eval_batch_size", "gradient_accumulation_steps"]:
        print(f"  {k}: {config.get(k)} ({type(config.get(k)).__name__})")

    print(f"\nLoading data from {config['data_dir']}...")
    train_records = load_dataset(config['data_dir'], 'train')
    valid_records = load_dataset(config['data_dir'], 'valid')
    test_records = load_dataset(config['data_dir'], 'test')

    if args.smoke_test:
        n_smoke = 32
        train_records = train_records[:n_smoke]
        valid_records = valid_records[:n_smoke]
        test_records = test_records[:n_smoke]
        print(f"[SMOKE TEST] Using {n_smoke} records per split")

    print(f"  Train: {len(train_records):,} records")
    print(f"  Valid: {len(valid_records):,} records")
    print(f"  Test:  {len(test_records):,} records")

    # BioRecord access sanity check
    assert safe_get(test_records[0], "id") is not None, "test_records[0] has no id"
    assert safe_get(test_records[0], "text") is not None, "test_records[0] has no text"
    assert safe_get(test_records[0], "tokens") is not None, "test_records[0] has no tokens"
    assert safe_get(test_records[0], "bio_tags") is not None, "test_records[0] has no bio_tags"
    print("BioRecord access sanity check passed.")

    models_to_run = args.models or list(config['models'].keys())
    label2id = {'O': 0, 'B-COMP': 1, 'I-COMP': 2}
    id2label = {v: k for k, v in label2id.items()}
    all_results = []

    for model_key in models_to_run:
        if model_key not in config['models']:
            print(f"Unknown model: {model_key}")
            continue

        model_config = config['models'][model_key]
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        print(f"  Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"  Tokenizer is_fast: {getattr(tokenizer, 'is_fast', False)}")

        print(f"  Tokenizing data...")
        max_len = int(config.get('max_length', 256))

        # Sanity check on first 3 records
        encode_tokens_with_labels_sanity_check(
            tokenizer, train_records, label2id, max_len, n_samples=3)

        train_ds = prepare_hf_dataset(train_records, tokenizer, label2id, max_len)
        valid_ds = prepare_hf_dataset(valid_records, tokenizer, label2id, max_len)
        test_ds  = prepare_hf_dataset(test_records,  tokenizer, label2id, max_len)

        try:
            success = train_model(
                model_key=model_key,
                model_config=model_config,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                test_records=test_records,
                output_dir=output_dir,
                config=config,
                device=device,
                label2id=label2id,
                id2label=id2label,
                smoke_test=args.smoke_test,
                force_rerun=args.force_rerun,
            )
        except Exception as e:
            print(f"  [FATAL] Training failed: {e}")
            continue

        if success:
            run_dir = os.path.join(output_dir, model_key)
            metrics_path = os.path.join(run_dir, 'test_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    result = json.load(f)
                result['model_key'] = model_key
                result['model_name'] = model_config['model_name']
                all_results.append(result)

    # Save experiment results CSV
    if all_results:
        csv_path = os.path.join(output_dir, 'experiment_results.csv')
        keys = ['model_key', 'model_name', 'loss', 'num_train_epochs',
                'entity_precision', 'entity_recall', 'entity_f1',
                'token_f1_macro', 'token_f1_weighted', 'token_accuracy']
        with open(csv_path, 'w', newline='') as f:
            import csv as csvmod
            writer = csvmod.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, '') for k in keys})

        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*60}")
        hdr = (f"{'Model':<25} {'Loss':<12} {'Eps':>4} "
               f"{'Entity-F1':>11} {'Tok-F1':>9} {'Acc':>8}")
        print(hdr)
        print('-' * len(hdr))
        for r in sorted(all_results, key=lambda x: x.get('entity_f1', 0), reverse=True):
            print(f"{r['model_key']:<25} {r.get('loss','?'):<12} "
                  f"{r.get('num_train_epochs','?'):>4} "
                  f"{r.get('entity_f1',0):>11.4f} "
                  f"{r.get('token_f1_macro',0):>9.4f} "
                  f"{r.get('token_accuracy',0):>8.4f}")
        print(f"\nResults saved to {csv_path}")

    # Final size summary
    print(f"\n{'='*60}")
    print("OUTPUT SIZE SUMMARY")
    print(f"{'='*60}")
    for model_key in models_to_run:
        if model_key not in config['models']:
            continue
        run_dir = os.path.join(output_dir, model_key)
        if os.path.exists(run_dir):
            size = get_output_size(run_dir)
            print(f"  {model_key:<25}: {size['total_mb']:>8.2f} MB  "
                  f"({size['all_files']} files, "
                  f"{size['checkpoint_folders']} ckpts, "
                  f"{size['weight_files']} weights)")

    print("\nDone.")


if __name__ == '__main__':
    main()
