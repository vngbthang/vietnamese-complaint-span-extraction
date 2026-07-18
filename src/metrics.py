"""Evaluation metrics for BIO token classification and span-level extraction."""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report as sklearn_report,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------
# seqeval fallbacks — avoid hard dependency on seqeval package
# ---------------------------------------------------------------

def _get_seqeval_functions():
    """
    Lazy-load seqeval if available. Return (precision, recall, f1, classification_report).
    Falls back to manual sklearn-based implementations.
    """
    try:
        from seqeval.metrics import (
            classification_report as seqeval_report,
            f1_score as seqeval_f1,
            precision_score as seqeval_precision,
            recall_score as seqeval_recall,
        )
        return seqeval_precision, seqeval_recall, seqeval_f1, seqeval_report
    except (ImportError, OSError):
        pass

    def _seqeval_precision(y_true, y_pred, **kwargs):
        labels = kwargs.get('labels', None)
        average = kwargs.get('average', 'micro')
        tp = fp = fn = 0
        all_labels = set()
        for seq_t, seq_p in zip(y_true, y_pred):
            for t, p in zip(seq_t, seq_p):
                all_labels.add(t)
                if t == p:
                    tp += 1
                else:
                    if t != 'O':
                        fn += 1
                    if p != 'O':
                        fp += 1
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def _seqeval_recall(y_true, y_pred, **kwargs):
        average = kwargs.get('average', 'micro')
        tp = fp = fn = 0
        for seq_t, seq_p in zip(y_true, y_pred):
            for t, p in zip(seq_t, seq_p):
                if t == p:
                    tp += 1
                else:
                    if t != 'O':
                        fn += 1
                    if p != 'O':
                        fp += 1
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def _seqeval_f1(y_true, y_pred, **kwargs):
        p = _seqeval_precision(y_true, y_pred, **kwargs)
        r = _seqeval_recall(y_true, y_pred, **kwargs)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def _seqeval_report(y_true, y_pred, **kwargs):
        labels = kwargs.get('labels', [])
        digits = kwargs.get('digits', 4)
        output_dict = kwargs.get('output_dict', False)
        report = sklearn_report(y_true, y_pred, labels=labels, digits=digits, output_dict=output_dict)
        return report

    return _seqeval_precision, _seqeval_recall, _seqeval_f1, _seqeval_report


_seqeval_precision, _seqeval_recall, _seqeval_f1, _seqeval_report = _get_seqeval_functions()


def seqeval_precision(y_true, y_pred, **kwargs):
    return _seqeval_precision(y_true, y_pred, **kwargs)


def seqeval_recall(y_true, y_pred, **kwargs):
    return _seqeval_recall(y_true, y_pred, **kwargs)


def seqeval_f1(y_true, y_pred, **kwargs):
    return _seqeval_f1(y_true, y_pred, **kwargs)


def seqeval_report(y_true, y_pred, **kwargs):
    return _seqeval_report(y_true, y_pred, **kwargs)


# ---------------------------------------------------------------
# Span-level metrics
# ---------------------------------------------------------------

def bio_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert BIO tags to a list of (start_idx, end_idx, entity_type) spans.
    Spans are token-level, half-open interval [start, end).
    """
    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith('B-'):
            entity_type = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f'I-{entity_type}':
                j += 1
            spans.append((i, j, entity_type))
            i = j
        else:
            i += 1
    return spans


def extract_entity_spans(
    predictions: List[str],
    ground_truths: List[str],
) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    """Extract entity spans from BIO tag sequences."""
    pred_spans = bio_to_spans(predictions)
    gt_spans = bio_to_spans(ground_truths)
    return pred_spans, gt_spans


def span_level_metrics(
    pred_spans: List[Tuple[int, int, str]],
    gt_spans: List[Tuple[int, int, str]],
) -> Dict[str, float]:
    """Compute strict entity-level precision, recall, F1."""
    pred_set = set(pred_spans)
    gt_set = set(gt_spans)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'entity_precision': precision,
        'entity_recall': recall,
        'entity_f1': f1,
        'entity_tp': tp,
        'entity_fp': fp,
        'entity_fn': fn,
    }


# ---------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------

def token_level_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute token-level classification metrics using seqeval."""
    if labels is None:
        labels = sorted(set(t for seq in y_true for t in seq) |
                        set(t for seq in y_pred for t in seq))

    metrics = {
        'token_precision_macro': seqeval_precision(y_true, y_pred, average='macro'),
        'token_recall_macro': seqeval_recall(y_true, y_pred, average='macro'),
        'token_f1_macro': seqeval_f1(y_true, y_pred, average='macro'),
        'token_precision_weighted': seqeval_precision(y_true, y_pred, average='weighted'),
        'token_recall_weighted': seqeval_recall(y_true, y_pred, average='weighted'),
        'token_f1_weighted': seqeval_f1(y_true, y_pred, average='weighted'),
        'token_accuracy': seqeval_f1(y_true, y_pred, average=None).__class__(1.0)
                          if y_true == y_pred else accuracy_score(
                              [t for seq in y_true for t in seq],
                              [t for seq in y_pred for t in seq],
                          ),
    }

    # Per-label metrics
    report = seqeval_report(y_true, y_pred, digits=4, output_dict=True)
    per_label = {}
    for label in labels:
        if label in report:
            per_label[label] = {
                'precision': report[label].get('precision', 0.0),
                'recall': report[label].get('recall', 0.0),
                'f1': report[label].get('f1', 0.0),
                'support': report[label].get('support', 0),
            }
    metrics['per_label'] = per_label

    return metrics


def compute_all_metrics(
    all_gt_tags: List[List[str]],
    all_pred_tags: List[List[str]],
    label_list: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive metrics for BIO tagging."""
    pred_spans_list = [bio_to_spans(p) for p in all_pred_tags]
    gt_spans_list = [bio_to_spans(g) for g in all_gt_tags]

    # Aggregate span-level
    all_pred_flat = []
    all_gt_flat = []
    for ps, gs in zip(pred_spans_list, gt_spans_list):
        all_pred_flat.extend(ps)
        all_gt_flat.extend(gs)

    span_metrics = span_level_metrics(all_pred_flat, all_gt_flat)
    token_metrics = token_level_metrics(all_gt_tags, all_pred_tags, label_list)

    return {**span_metrics, **token_metrics}


def save_per_label_report(
    metrics: Dict[str, Any],
    output_path: str,
) -> None:
    """Save per-label metrics to a CSV file."""
    per_label = metrics.get('per_label', {})
    if not per_label:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('label,precision,recall,f1,support\n')
        for label, vals in sorted(per_label.items()):
            f.write(f'{label},{vals["precision"]:.4f},{vals["recall"]:.4f},'
                    f'{vals["f1"]:.4f},{vals["support"]}\n')


def save_error_analysis(
    records: List[Dict[str, Any]],
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    output_path: str,
) -> None:
    """Save error analysis to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for rec, preds, gts in zip(records, predictions, ground_truths):
        pred_spans = set(bio_to_spans(preds))
        gt_spans = set(bio_to_spans(gts))

        missing = gt_spans - pred_spans
        extra = pred_spans - gt_spans

        rows.append({
            'id': rec.get('id', ''),
            'source': rec.get('source', ''),
            'split': rec.get('split', ''),
            'num_gt_spans': len(gt_spans),
            'num_pred_spans': len(pred_spans),
            'correct': int(pred_spans == gt_spans),
            'missing_spans': ' | '.join(f'[{s}:{e}:{t}]' for s, e, t in missing) or '',
            'extra_spans': ' | '.join(f'[{s}:{e}:{t}]' for s, e, t in extra) or '',
            'text': rec.get('text', '')[:200],
        })

    keys = ['id', 'source', 'split', 'num_gt_spans', 'num_pred_spans',
            'correct', 'missing_spans', 'extra_spans', 'text']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        import csv as csvmod
        writer = csvmod.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions_jsonl(
    records: List[Dict[str, Any]],
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    output_path: str,
) -> None:
    """Save predictions as JSONL for later analysis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec, preds, gts in zip(records, predictions, ground_truths):
            out = {
                'id': rec.get('id', ''),
                'source': rec.get('source', ''),
                'text': rec.get('text', ''),
                'tokens': rec.get('tokens', []),
                'bio_tags_true': gts,
                'bio_tags_pred': preds,
                'spans_true': bio_to_spans(gts),
                'spans_pred': bio_to_spans(preds),
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
