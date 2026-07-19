"""Error analysis utilities for BIO token classification."""

import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _record_field(record: Any, key: str, default: Any = None) -> Any:
    """Read a field from a record, supporting dict or dataclass/object."""
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def analyze_token_errors(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    tokens: List[List[str]],
    ids: List[str],
) -> List[Dict[str, Any]]:
    """Analyze token-level prediction errors."""
    errors = []
    for i, (true_seq, pred_seq, tok_seq, rec_id) in enumerate(zip(y_true, y_pred, tokens, ids)):
        for j, (t, p, tok) in enumerate(zip(true_seq, pred_seq, tok_seq)):
            if t != p:
                errors.append({
                    'record_id': rec_id,
                    'token_idx': j,
                    'token': tok,
                    'true_tag': t,
                    'pred_tag': p,
                    'is_error': True,
                    'label_type': 'O' if t == 'O' else 'ASP' if 'ASP' in t else 'COMP',
                })
    return errors


def summarize_error_types(errors: List[Dict[str, Any]]) -> Dict[str, int]:
    """Summarize error types."""
    summary = Counter()
    for err in errors:
        t = err['true_tag']
        p = err['pred_tag']
        summary[f'{t}->{p}'] += 1
    return dict(summary.most_common())


def error_distribution_by_label(
    errors: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Show error distribution by true label."""
    by_true = defaultdict(Counter)
    for err in errors:
        by_true[err['true_tag']][err['pred_tag']] += 1
    return {k: dict(v) for k, v in by_true.items()}


def span_error_analysis(
    pred_spans: List[Tuple[int, int, str]],
    gt_spans: List[Tuple[int, int, str]],
) -> Dict[str, Any]:
    """Detailed analysis of span-level errors."""
    pred_set = set(pred_spans)
    gt_set = set(gt_spans)

    missing = gt_set - pred_set
    extra = pred_set - gt_set

    # Partial errors: spans that overlap but don't match
    partial_errors = []
    for p in pred_set:
        for g in gt_set:
            if p[2] == g[2]:  # Same type
                if not (p[0] >= g[1] or g[0] >= p[1]):  # Overlap
                    if p != g:
                        partial_errors.append({
                            'type': p[2],
                            'pred': p,
                            'gt': g,
                            'pred_len': p[1] - p[0],
                            'gt_len': g[1] - g[0],
                        })

    return {
        'missing_count': len(missing),
        'extra_count': len(extra),
        'partial_overlap_count': len(partial_errors),
        'missing_by_type': Counter(s[2] for s in missing),
        'extra_by_type': Counter(s[2] for s in extra),
        'partial_errors_sample': partial_errors[:10],
    }


def save_full_analysis(
    errors: List[Dict[str, Any]],
    error_summary: Dict[str, int],
    output_path: str,
) -> None:
    """Save full error analysis to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_errors': len(errors),
            'error_types': error_summary,
            'sample_errors': errors[:100],
        }, f, indent=2, ensure_ascii=False)


def save_per_record_analysis(
    records: List[Any],
    y_true: List[List[str]],
    y_pred: List[List[str]],
    output_path: str,
) -> None:
    """Save per-record error analysis with id, gold_spans, pred_spans, error_type, note.

    Works with both dict records and BioRecord dataclass objects.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    keys = ['id', 'text', 'gold_spans', 'pred_spans', 'error_type', 'note']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for rec, true_seq, pred_seq in zip(records, y_true, y_pred):
            true_spans = bio_to_spans_inline(true_seq)
            pred_spans = bio_to_spans_inline(pred_seq)

            if true_spans == pred_spans:
                error_type = 'none'
                note = 'perfect'
            elif not true_spans:
                error_type = 'extra'
                note = f'{len(pred_spans)} extra span(s)'
            elif not pred_spans:
                error_type = 'missing'
                note = f'{len(true_spans)} missing span(s)'
            else:
                missing = set(true_spans) - set(pred_spans)
                extra = set(pred_spans) - set(true_spans)
                error_type = 'partial' if (missing or extra) else 'none'
                note = f'missing={len(missing)}, extra={len(extra)}'

            text_val = _record_field(rec, 'text', '') or ''
            if not isinstance(text_val, str):
                text_val = str(text_val)

            writer.writerow({
                'id': _record_field(rec, 'id', ''),
                'text': text_val[:200],
                'gold_spans': ' | '.join(f'[{s}:{e}:{t}]' for s, e, t in true_spans),
                'pred_spans': ' | '.join(f'[{s}:{e}:{t}]' for s, e, t in pred_spans),
                'error_type': error_type,
                'note': note,
            })


def bio_to_spans_inline(seq: List[str]) -> List[Tuple[int, int, str]]:
    """Convert a BIO tag sequence to (start, end, type) tuples."""
    spans = []
    j = 0
    while j < len(seq):
        tag = seq[j]
        if tag.startswith('B-'):
            label = tag[2:]
            s = j
            j += 1
            while j < len(seq) and seq[j] == f'I-{label}':
                j += 1
            spans.append((s, j, label))
        else:
            j += 1
    return spans


def find_boundary_errors(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    tokens: List[List[str]],
    ids: List[str],
    offsets: Optional[List[List[Dict[str, int]]]] = None,
) -> List[Dict[str, Any]]:
    """Find errors at span boundaries (start/end token mismatches)."""
    boundary_errors = []
    for i, (true_seq, pred_seq, tok_seq, rec_id) in enumerate(zip(y_true, y_pred, tokens, ids)):

        def get_spans(seq):
            spans = []
            j = 0
            while j < len(seq):
                if seq[j].startswith('B-'):
                    label = seq[j][2:]
                    s = j
                    j += 1
                    while j < len(seq) and seq[j] == f'I-{label}':
                        j += 1
                    spans.append((s, j, label))
                else:
                    j += 1
            return spans

        true_spans = get_spans(true_seq)
        pred_spans = get_spans(pred_seq)

        for ts in true_spans:
            matching_pred = [p for p in pred_spans if p[2] == ts[2]]
            if not matching_pred:
                boundary_errors.append({
                    'id': rec_id,
                    'error_type': 'missing_span',
                    'true_span': ts,
                    'true_tokens': ' '.join(tok_seq[ts[0]:ts[1]]),
                    'pred_tokens': '',
                })
            else:
                for ps in matching_pred:
                    if ps[0] != ts[0]:
                        boundary_errors.append({
                            'id': rec_id,
                            'error_type': 'boundary_start_mismatch',
                            'true_span': ts,
                            'pred_span': ps,
                            'true_tokens': ' '.join(tok_seq[ts[0]:ts[1]]),
                            'pred_tokens': ' '.join(tok_seq[ps[0]:ps[1]]) if ps[1] <= len(tok_seq) else '?',
                        })
                    if ps[1] != ts[1]:
                        boundary_errors.append({
                            'id': rec_id,
                            'error_type': 'boundary_end_mismatch',
                            'true_span': ts,
                            'pred_span': ps,
                            'true_tokens': ' '.join(tok_seq[ts[0]:ts[1]]),
                            'pred_tokens': ' '.join(tok_seq[ps[0]:ps[1]]) if ps[1] <= len(tok_seq) else '?',
                        })

    return boundary_errors
