#!/usr/bin/env python3
"""
Data integrity checker for complaint span extraction project.

Checks both:
- data/complaint_span_bio_clean/ (main: O, B-COMP, I-COMP)
- data/auxiliary_ate_bio_clean/   (aux:  O, B-ASP, I-ASP)

Exit code: 0 if all checks pass, non-zero otherwise.
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    dir: str
    label_set: Set[str]
    splits: List[str]


DATASETS = [
    DatasetConfig(
        name='complaint_span_bio_clean',
        dir='data/complaint_span_bio_clean',
        label_set={'O', 'B-COMP', 'I-COMP'},
        splits=['train', 'valid', 'test'],
    ),
    DatasetConfig(
        name='auxiliary_ate_bio_clean',
        dir='data/auxiliary_ate_bio_clean',
        label_set={'O', 'B-ASP', 'I-ASP'},
        splits=['train', 'valid', 'test'],
    ),
]

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def load_records(split_dir: str) -> List[Dict]:
    records = []
    with open(split_dir, encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def derive_spans_from_bio(bio_tags: List[str],
                           offsets: List) -> List[Tuple[int, int]]:
    """Derive token-level spans from BIO tags."""
    spans = []
    i = 0
    while i < len(bio_tags):
        if bio_tags[i].startswith('B-'):
            prefix = bio_tags[i][2:]
            s_start = i
            j = i + 1
            while j < len(bio_tags) and bio_tags[j] == f'I-{prefix}':
                j += 1
            spans.append((s_start, j))
            i = j
        else:
            i += 1
    return spans


# ---------------------------------------------------------------
# Checks
# ---------------------------------------------------------------

def check_files_exist(config: DatasetConfig) -> Tuple[bool, str, Dict]:
    for split in config.splits:
        path = os.path.join(config.dir, f'{split}.jsonl')
        if not os.path.exists(path):
            return False, f"Missing file: {path}", {}
    return True, "OK", {}


def check_schema_and_labels(config: DatasetConfig) -> Tuple[bool, str, Dict]:
    stats = defaultdict(int)
    all_tags = set()
    errors = []

    for split in config.splits:
        path = os.path.join(config.dir, f'{split}.jsonl')
        records = load_records(path)

        for rec in records:
            # Required fields
            for field in ['id', 'text', 'tokens', 'token_offsets', 'bio_tags']:
                if field not in rec:
                    errors.append(f"[{rec.get('id','?')}] Missing field: {field}")
                    continue

            # Field lengths
            n_tok = len(rec['tokens'])
            n_off = len(rec['token_offsets'])
            n_bio = len(rec['bio_tags'])
            if not (n_tok == n_off == n_bio):
                errors.append(
                    f"[{rec['id']}] Length mismatch: "
                    f"tokens={n_tok}, offsets={n_off}, bio={n_bio}"
                )

            # Label set
            for tag in rec['bio_tags']:
                all_tags.add(tag)

            # BIO validity: I- without preceding B-
            for i, tag in enumerate(rec['bio_tags']):
                if tag.startswith('I-'):
                    if i == 0:
                        errors.append(f"[{rec['id']}] I- at position 0")
                    elif not rec['bio_tags'][i-1].startswith('B-') and \
                            not rec['bio_tags'][i-1].startswith('I-'):
                        errors.append(
                            f"[{rec['id']}] I- not preceded by B- or I-: "
                            f"pos={i}, prev={rec['bio_tags'][i-1]}"
                        )

            stats[f'{split}_records'] += 1
            stats[f'{split}_tokens'] += n_tok
            stats[f'{split}_bio_sum'] += n_bio

    unknown_tags = all_tags - config.label_set
    if unknown_tags:
        errors.append(f"Unknown tags: {unknown_tags} (expected: {config.label_set})")

    ok = len(errors) == 0
    msg = "OK" if ok else f"FAILED ({len(errors)} errors)"
    return ok, msg, dict(stats)


def check_bio_token_sum(config: DatasetConfig) -> Tuple[bool, str, Dict]:
    errors = []
    totals = {'tokens': 0, 'bio_sum': 0}
    tag_totals = defaultdict(int)

    for split in config.splits:
        path = os.path.join(config.dir, f'{split}.jsonl')
        records = load_records(path)

        for rec in records:
            n_tok = len(rec['tokens'])
            n_bio = len(rec['bio_tags'])

            if n_tok != n_bio:
                errors.append(f"[{rec['id']}] tokens={n_tok} != bio_tags={n_bio}")

            totals['tokens'] += n_tok
            totals['bio_sum'] += n_bio

            for tag in rec['bio_tags']:
                tag_totals[tag] += 1

    # Check O + B-XXX + I-XXX = total
    bio_sum_computed = sum(tag_totals.values())
    if bio_sum_computed != totals['tokens']:
        errors.append(
            f"BIO sum mismatch: O+B+I={bio_sum_computed} != total_tokens={totals['tokens']}"
        )

    ok = len(errors) == 0
    msg = "OK" if ok else f"FAILED ({len(errors)} errors)"
    return ok, msg, {'total_tokens': totals['tokens'],
                     'bio_sum': totals['bio_sum'],
                     'tag_dist': dict(tag_totals)}


def check_roundtrip_bio_spans(config: DatasetConfig) -> Tuple[bool, str, Dict]:
    """Verify that BIO-derived spans roundtrip without errors."""
    errors = []
    total_spans = 0

    for split in config.splits:
        path = os.path.join(config.dir, f'{split}.jsonl')
        records = load_records(path)

        for rec in records:
            bio_tags = rec['bio_tags']
            offsets = rec['token_offsets']
            spans = derive_spans_from_bio(bio_tags, offsets)
            total_spans += len(spans)

            for (s, e) in spans:
                if s >= e:
                    errors.append(f"[{rec['id']}] Invalid span: [{s}:{e}]")
                if e > len(bio_tags):
                    errors.append(
                        f"[{rec['id']}] Span out of bounds: [{s}:{e}] > {len(bio_tags)}"
                    )

    ok = len(errors) == 0
    msg = "OK" if ok else f"FAILED ({len(errors)} errors)"
    return ok, msg, {'total_spans': total_spans, 'errors': len(errors)}


def check_split_leakage(config: DatasetConfig) -> Tuple[bool, str, Dict]:
    """Check that no record IDs appear in multiple splits."""
    id_sets = {}
    all_ids = set()

    for split in config.splits:
        path = os.path.join(config.dir, f'{split}.jsonl')
        records = load_records(path)
        ids = set(rec['id'] for rec in records)
        id_sets[split] = ids

    duplicates = []
    for i, s1 in enumerate(config.splits):
        for s2 in config.splits[i+1:]:
            dup = id_sets[s1] & id_sets[s2]
            if dup:
                duplicates.extend(list(dup)[:10])  # limit output

    ok = len(duplicates) == 0
    msg = "OK" if ok else f"FAILED ({len(duplicates)} duplicate IDs)"
    return ok, msg, {
        'train_ids': len(id_sets.get('train', set())),
        'valid_ids': len(id_sets.get('valid', set())),
        'test_ids': len(id_sets.get('test', set())),
        'duplicates': len(duplicates),
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("DATA INTEGRITY CHECK")
    print("=" * 60)

    all_passed = True

    for config in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {config.name}")
        print(f"Expected labels: {sorted(config.label_set)}")
        print(f"{'='*60}")

        # 1. File existence
        ok, msg, _ = check_files_exist(config)
        print(f"  [1] Files exist:           {msg}")
        if not ok:
            all_passed = False
            continue

        # 2. Schema and labels
        ok, msg, stats = check_schema_and_labels(config)
        print(f"  [2] Schema & labels:       {msg}")
        if not ok:
            all_passed = False
        for k, v in sorted(stats.items()):
            print(f"      {k}: {v:,}" if isinstance(v, int) else f"      {k}: {v}")

        # 3. BIO token sum
        ok, msg, stats = check_bio_token_sum(config)
        print(f"  [3] BIO sum check:         {msg}")
        if not ok:
            all_passed = False
        print(f"      total_tokens: {stats['total_tokens']:,}")
        print(f"      bio_sum:      {stats['bio_sum']:,}")
        for tag, cnt in sorted(stats['tag_dist'].items()):
            pct = cnt / stats['total_tokens'] * 100
            print(f"      {tag:8s}: {cnt:8,} ({pct:.2f}%)")

        # 4. Roundtrip BIO spans
        ok, msg, stats = check_roundtrip_bio_spans(config)
        print(f"  [4] BIO roundtrip:         {msg}")
        if not ok:
            all_passed = False
        print(f"      total_spans: {stats['total_spans']:,}")
        print(f"      errors:      {stats['errors']}")

        # 5. Split leakage
        ok, msg, stats = check_split_leakage(config)
        print(f"  [5] Split leakage:         {msg}")
        if not ok:
            all_passed = False
        print(f"      train_ids: {stats['train_ids']:,}")
        print(f"      valid_ids: {stats['valid_ids']:,}")
        print(f"      test_ids:  {stats['test_ids']:,}")
        print(f"      duplicates: {stats['duplicates']}")

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("RESULT: SOME CHECKS FAILED")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
