#!/usr/bin/env python3
"""
Transfer learning training for complaint span extraction.

NO CHECKPOINTS ARE SAVED.
Only metrics, predictions, and reports are written to disk.

Transfer strategy: pretrain encoder on auxiliary ATE data (in memory),
then copy encoder weights into a new complaint model (in memory),
then fine-tune on ViOCD-Span.
"""

import argparse
import copy
import glob
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from datasets import Dataset
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_utils import load_split, filter_by_source, BioRecord
from src.metrics import bio_to_spans, save_error_analysis, save_predictions_jsonl

# ---------------------------------------------------------------
# Cleanup utilities
# ---------------------------------------------------------------

def cleanup_checkpoints(output_dir: str) -> Dict[str, int]:
    """Remove all checkpoint folders and model weight files."""
    removed = {'checkpoint_folders': 0, 'weight_files': 0}
    if not os.path.exists(output_dir):
        return removed

    for pattern in [
        os.path.join(output_dir, 'checkpoint-*'),
        os.path.join(output_dir, '**', 'checkpoint-*'),
    ]:
        for ckpt_dir in glob.glob(pattern, recursive=True):
            if os.path.isdir(ckpt_dir):
                os.system(f'rm -rf {repr(ckpt_dir)}')
                removed['checkpoint_folders'] += 1

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
            checkpoint_folders += sum(1 for d in dirs if d.startswith('checkpoint-'))
            for f in files:
                all_files += 1
                fpath = os.path.join(root, f)
                try:
                    total_size += os.path.getsize(fpath)
                except OSError:
                    pass
                if any(f.endswith(ext) or f == ext for ext in [
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
    parser = argparse.ArgumentParser(
        description='Transfer learning training (no checkpoints)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--strategy', type=str,
                        choices=['aux_all_then_complaint',
                                 'causasent_then_complaint',
                                 'uvisd4sa_then_complaint'],
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

def prepare_dataset(records, tokenizer, label2id, max_length: int):
    """Prepare HuggingFace Dataset from BioRecords."""
    texts = [' '.join(r.tokens) for r in records]
    dataset_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}

    batch_size = 500
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            return_tensors=None,
        )
        batch_wids = batch_enc.word_ids()

        for i in range(len(batch_texts)):
            word_ids = [batch_wids[j] for j in range(len(batch_enc['input_ids'][i]))]
            rec = records[batch_start + i]
            labels = []
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                elif wid < len(rec.bio_tags):
                    tag = rec.bio_tags[wid]
                    labels.append(label2id.get(tag, 0))
                else:
                    labels.append(label2id.get('O', 0))
            dataset_dict['input_ids'].append(batch_enc['input_ids'][i])
            dataset_dict['attention_mask'].append(batch_enc['attention_mask'][i])
            dataset_dict['labels'].append(labels)

    return Dataset.from_dict(dataset_dict)


# ---------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_labels: int,
                label2id: Dict, id2label: Dict, device: torch.device):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True)
    model.to(device)
    return model


def extract_encoder_state_dict(model: AutoModelForTokenClassification) -> Dict:
    """Extract encoder/base model state dict for in-memory transfer."""
    state = {}
    for key, value in model.state_dict().items():
        if not key.startswith('classifier'):
            state[key] = value.clone()
    return state


def load_encoder_into_model(model: AutoModelForTokenClassification,
                             encoder_state: Dict) -> None:
    """
    Load encoder weights in-place. Only loads keys that match shape.
    The classifier (head) remains initialized.
    """
    current_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in encoder_state.items():
        if key in current_state:
            if current_state[key].shape == value.shape:
                current_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key}: shape mismatch "
                                   f"{value.shape} vs {current_state[key].shape}")
        else:
            skipped_keys.append(f"{key}: not in target model")

    model.load_state_dict(current_state, strict=False)
    print(f"  Loaded {len(loaded_keys)} encoder parameters in-memory")
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} keys (head layers or shape mismatches)")


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

        flat_true = [t for seq in true_tags for t in seq]
        flat_pred = [p for seq in pred_tags for p in seq]
        token_acc = (sum(1 for a, b in zip(flat_true, flat_pred) if a == b)
                      / len(flat_true) if flat_true else 0.0)

        return {
            'entity_precision': span_p,
            'entity_recall': span_r,
            'entity_f1': span_f1,
            'token_f1_macro': token_f1,
            'token_precision_macro': token_p,
            'token_recall_macro': token_r,
            'token_accuracy': token_acc,
        }
    return compute_metrics


# ---------------------------------------------------------------
# Training pipeline (no checkpoints)
# ---------------------------------------------------------------

def train_transfer(
    strategy_key: str,
    strategy_config: Dict,
    config: Dict,
    device: torch.device,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> bool:
    """Run complete transfer learning strategy in memory, no disk checkpoints."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_key}")
    print(f"{'='*60}")
    desc = strategy_config.get('description', '')
    if desc:
        print(f"  {desc}")
    print(f"  NOTE: No checkpoints will be saved to disk.")

    run_dir = os.path.join(config['output_dir'], strategy_key)
    os.makedirs(run_dir, exist_ok=True)

    if not force_rerun and os.path.exists(os.path.join(run_dir, 'completed_result.json')):
        print(f"  [SKIP] Already trained.")
        return True

    model_name = config['model_name']
    max_length = config.get('max_length', 256)

    # ---- Load auxiliary data ----
    aux_records_all = []
    for split in ['train', 'valid']:
        path = os.path.join(config['aux_data_dir'], f'{split}.jsonl')
        if os.path.exists(path):
            aux_records_all.extend(load_split(config['aux_data_dir'], split))

    aux_filter = strategy_config.get('aux_filter_source')
    if aux_filter:
        aux_records_all = filter_by_source(aux_records_all, aux_filter)
        print(f"  Filtered to source={aux_filter}: {len(aux_records_all):,} records")

    if smoke_test:
        aux_records_all = aux_records_all[:100]

    print(f"  Auxiliary records: {len(aux_records_all):,}")

    aux_label2id = {'O': 0, 'B-ASP': 1, 'I-ASP': 2}
    aux_id2label = {v: k for k, v in aux_label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split aux for validation
    n_aux = len(aux_records_all)
    n_aux_train = int(n_aux * 0.9)
    aux_train = aux_records_all[:n_aux_train]
    aux_val = aux_records_all[n_aux_train:]

    print(f"  Aux train: {len(aux_train):,}, aux val: {len(aux_val):,}")
    aux_train_ds = prepare_dataset(aux_train, tokenizer, aux_label2id, max_length)
    aux_val_ds = prepare_dataset(aux_val, tokenizer, aux_label2id, max_length)

    aux_epochs = 1 if smoke_test else config.get('aux_epochs', 3)

    aux_args = TrainingArguments(
        output_dir=os.path.join(run_dir, 'phase1_aux'),
        num_train_epochs=aux_epochs,
        per_device_train_batch_size=(4 if smoke_test
                                      else config.get('train_batch_size', 16)),
        per_device_eval_batch_size=config.get('eval_batch_size', 16),
        learning_rate=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.01),
        warmup_ratio=config.get('warmup_ratio', 0.06),
        fp16=config.get('fp16', True) and device.type == 'cuda',
        save_strategy='no',
        save_total_limit=0,
        load_best_model_at_end=False,
        save_safetensors=False,
        logging_dir=os.path.join(run_dir, 'phase1_aux', 'logs'),
        logging_steps=20,
        eval_strategy='epoch' if not smoke_test else 'no',
        report_to='none',
        seed=config.get('seed', 42),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    # ---- Phase 1: Auxiliary pretraining (in memory) ----
    print(f"\n  === Phase 1: Auxiliary ATE Pretraining (in memory) ===")

    aux_model = build_model(model_name, len(aux_label2id),
                              aux_label2id, aux_id2label, device)

    aux_trainer = Trainer(
        model=aux_model,
        args=aux_args,
        train_dataset=aux_train_ds,
        eval_dataset=aux_val_ds if not smoke_test else None,
        compute_metrics=compute_metrics_fn(aux_id2label) if not smoke_test else None,
        data_collator=DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=8),
    )

    start_aux = time.time()
    try:
        aux_trainer.train()
    except Exception as e:
        print(f"  [ERROR] Auxiliary training failed: {e}")
        del aux_model
        del aux_trainer
        torch.cuda.empty_cache()
        raise
    aux_time = time.time() - start_aux
    print(f"  Auxiliary training: {aux_time:.1f}s")

    # ---- Extract encoder weights IN MEMORY ----
    print(f"  Extracting encoder state dict in memory...")
    encoder_state = extract_encoder_state_dict(aux_model)

    # Free auxiliary model memory
    del aux_trainer
    del aux_model
    torch.cuda.empty_cache()
    print(f"  Auxiliary model freed from memory.")

    # ---- Load complaint data ----
    target_train = load_split(config['target_data_dir'], 'train')
    target_valid = load_split(config['target_data_dir'], 'valid')
    target_test  = load_split(config['target_data_dir'], 'test')

    if smoke_test:
        target_train = target_train[:32]
        target_valid = target_valid[:32]
        target_test  = target_test[:32]

    print(f"  Target train: {len(target_train):,}")
    print(f"  Target valid: {len(target_valid):,}")
    print(f"  Target test:  {len(target_test):,}")

    target_label2id = {'O': 0, 'B-COMP': 1, 'I-COMP': 2}
    target_id2label = {v: k for k, v in target_label2id.items()}

    # ---- Phase 2: Build complaint model and load encoder in memory ----
    print(f"\n  === Phase 2: Complaint Fine-tuning (pretrained encoder in memory) ===")

    target_model = build_model(
        model_name, len(target_label2id),
        target_label2id, target_id2label, device)

    # Load encoder weights in memory — classifier head stays fresh
    print(f"  Loading pretrained encoder into new complaint model (in memory)...")
    load_encoder_into_model(target_model, encoder_state)
    del encoder_state
    torch.cuda.empty_cache()

    target_epochs = 1 if smoke_test else config.get('target_epochs', 5)

    target_dir = os.path.join(run_dir, 'phase2_target')
    os.makedirs(target_dir, exist_ok=True)

    target_args = TrainingArguments(
        output_dir=target_dir,
        num_train_epochs=target_epochs,
        per_device_train_batch_size=(4 if smoke_test
                                      else config.get('train_batch_size', 16)),
        per_device_eval_batch_size=config.get('eval_batch_size', 16),
        learning_rate=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.01),
        warmup_ratio=config.get('warmup_ratio', 0.06),
        fp16=config.get('fp16', True) and device.type == 'cuda',
        save_strategy='no',
        save_total_limit=0,
        load_best_model_at_end=False,
        save_safetensors=False,
        logging_dir=os.path.join(target_dir, 'logs'),
        logging_steps=20,
        eval_strategy='epoch' if not smoke_test else 'no',
        report_to='none',
        seed=config.get('seed', 42),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    target_train_ds = prepare_dataset(target_train, tokenizer, target_label2id, max_length)
    target_valid_ds = prepare_dataset(target_valid, tokenizer, target_label2id, max_length)
    target_test_ds  = prepare_dataset(target_test,  tokenizer, target_label2id, max_length)

    target_trainer = Trainer(
        model=target_model,
        args=target_args,
        train_dataset=target_train_ds,
        eval_dataset=target_valid_ds if not smoke_test else None,
        compute_metrics=compute_metrics_fn(target_id2label) if not smoke_test else None,
        data_collator=DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=8),
    )

    start_target = time.time()
    try:
        target_trainer.train()
    except Exception as e:
        print(f"  [ERROR] Target fine-tuning failed: {e}")
        del target_model
        del target_trainer
        torch.cuda.empty_cache()
        raise
    target_time = time.time() - start_target
    print(f"  Target fine-tuning: {target_time:.1f}s")

    # ---- Evaluate ----
    print(f"  Evaluating on test set...")
    test_output = target_trainer.predict(target_test_ds)
    preds = np.argmax(test_output.predictions, axis=2)

    true_tags, pred_tags = [], []
    for seq_pred, seq_labels in zip(preds, test_output.label_ids):
        t_seq, p_seq = [], []
        for p, l in zip(seq_pred, seq_labels):
            if l != -100:
                t_seq.append(target_id2label.get(l, 'O'))
                p_seq.append(target_id2label.get(p, 'O'))
        true_tags.append(t_seq)
        pred_tags.append(p_seq)

    token_p = seqeval_precision(true_tags, pred_tags)
    token_r = seqeval_recall(true_tags, pred_tags)
    token_f1 = seqeval_f1(true_tags, pred_tags)
    flat_true = [t for seq in true_tags for t in seq]
    flat_pred = [p for seq in pred_tags for p in seq]
    token_acc = (sum(1 for a, b in zip(flat_true, flat_pred) if a == b)
                  / len(flat_true) if flat_true else 0.0)
    token_f1_w = seqeval_f1(true_tags, pred_tags, average='weighted')

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

    test_metrics = {
        'strategy': strategy_key,
        'model_name': model_name,
        'entity_precision': span_p,
        'entity_recall': span_r,
        'entity_f1': span_f1,
        'entity_tp': tp,
        'entity_fp': fp,
        'entity_fn': fn,
        'token_f1_macro': token_f1,
        'token_f1_weighted': token_f1_w,
        'token_precision_macro': token_p,
        'token_recall_macro': token_r,
        'token_accuracy': token_acc,
        'aux_time_seconds': aux_time,
        'target_time_seconds': target_time,
        'total_time_seconds': aux_time + target_time,
        'aux_epochs': aux_epochs,
        'target_epochs': target_epochs,
    }

    # Save predictions and reports
    pred_path = os.path.join(run_dir, 'test_predictions.jsonl')
    save_predictions_jsonl(target_test, pred_tags, true_tags, pred_path)
    err_path = os.path.join(run_dir, 'error_analysis.csv')
    save_error_analysis(target_test, pred_tags, true_tags, err_path)

    # Train log
    with open(os.path.join(run_dir, 'train_log.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Strategy: {strategy_key}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"NO_CHECKPOINTS_SAVED=True\n")
        f.write(f"Auxiliary epochs: {aux_epochs}\n")
        f.write(f"Target epochs: {target_epochs}\n")
        f.write(f"Auxiliary time: {aux_time:.1f}s\n")
        f.write(f"Target time: {target_time:.1f}s\n")
        f.write(f"Total time: {aux_time + target_time:.1f}s\n")

    test_metrics['prediction_path'] = pred_path
    test_metrics['error_analysis_path'] = err_path

    with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    with open(os.path.join(run_dir, 'completed_result.json'), 'w') as f:
        json.dump({'strategy': strategy_key, 'model_name': model_name,
                   'test_metrics': test_metrics}, f, indent=2)

    # Free model memory
    del target_trainer
    del target_model
    torch.cuda.empty_cache()

    # Cleanup checkpoints
    cleanup = config.get('cleanup_checkpoints', True)
    if cleanup:
        for subdir in [os.path.join(run_dir, 'phase1_aux'),
                       os.path.join(run_dir, 'phase2_target')]:
            if os.path.exists(subdir):
                removed = cleanup_checkpoints(subdir)
                if removed['checkpoint_folders'] > 0 or removed['weight_files'] > 0:
                    print(f"  [CLEANUP] {subdir}: removed {removed['checkpoint_folders']} "
                          f"ckpt folders, {removed['weight_files']} weight files")

    size = get_output_size(run_dir)
    print(f"  Output size: {size['total_mb']:.2f} MB "
          f"({size['all_files']} files, "
          f"{size['checkpoint_folders']} ckpts, "
          f"{size['weight_files']} weights)")
    print(f"  Results: Entity-F1={span_f1:.4f} "
          f"(P={span_p:.4f}, R={span_r:.4f})")
    print(f"  Results saved to {run_dir}")
    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(config.get('seed', 42))

    strategies = args.strategy or list(config['strategies'].keys())
    if isinstance(strategies, str):
        strategies = [strategies]

    all_results = []
    for strategy_key in strategies:
        if strategy_key not in config['strategies']:
            print(f"Unknown strategy: {strategy_key}")
            continue

        try:
            success = train_transfer(
                strategy_key=strategy_key,
                strategy_config=config['strategies'][strategy_key],
                config=config,
                device=device,
                smoke_test=args.smoke_test,
                force_rerun=args.force_rerun,
            )
        except Exception as e:
            print(f"  [FATAL] Strategy failed: {e}")
            continue

        if success:
            run_dir = os.path.join(config['output_dir'], strategy_key)
            metrics_path = os.path.join(run_dir, 'test_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    all_results.append({**json.load(f), 'strategy': strategy_key})

    if all_results:
        print(f"\n{'='*60}")
        print("TRANSFER LEARNING RESULTS")
        print(f"{'='*60}")
        hdr = f"{'Strategy':<35} {'Entity-F1':>11} {'Tok-F1':>9} {'Time':>10}"
        print(hdr)
        print('-' * len(hdr))
        for r in sorted(all_results, key=lambda x: x.get('entity_f1', 0), reverse=True):
            total_t = r.get('total_time_seconds', 0)
            print(f"{r['strategy']:<35} "
                  f"{r.get('entity_f1', 0):>11.4f} "
                  f"{r.get('token_f1_macro', 0):>9.4f} "
                  f"{total_t:>9.1f}s")

    # Final size summary
    print(f"\n{'='*60}")
    print("OUTPUT SIZE SUMMARY")
    print(f"{'='*60}")
    for strategy_key in (args.strategy or list(config['strategies'].keys())):
        if strategy_key not in config['strategies']:
            continue
        run_dir = os.path.join(config['output_dir'], strategy_key)
        if os.path.exists(run_dir):
            size = get_output_size(run_dir)
            print(f"  {strategy_key:<35}: {size['total_mb']:>8.2f} MB  "
                  f"({size['all_files']} files, "
                  f"{size['checkpoint_folders']} ckpts, "
                  f"{size['weight_files']} weights)")

    print("\nDone.")


if __name__ == '__main__':
    main()
