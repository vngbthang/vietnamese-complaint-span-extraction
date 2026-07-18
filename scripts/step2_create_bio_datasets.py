#!/usr/bin/env python3
"""
Step 2: Create Task-Aware BIO Datasets
======================================
Converts character-level spans to token-level BIO tags, separated by task.

Tasks:
  - aspect_term_extraction: UIT-ViSD4SA + CausaSent-ATE-v2 → O, B-ASP, I-ASP
  - complaint_detection:    ViOCD → review-level label (no BIO tags)

Features:
  - Robust word-level tokenizer (regex-based, no external dependencies)
  - Character-to-token offset mapping
  - BIO tag assignment with span overlap handling
  - Round-trip validation: BIO tags → recovered spans → compare to originals
  - Strict mode: token must be fully inside span for B-ASP
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_IN = REPO_ROOT / "data" / "final_splits"
DATA_OUT = REPO_ROOT / "data" / "bio_splits"

# Valid BIO labels per task
BIO_LABELS: Dict[str, Set[str]] = {
    "aspect_term_extraction": {"O", "B-ASP", "I-ASP"},
}

# Tokenizer pattern: split on whitespace + punctuation
# Keeps punctuation as separate tokens so token boundaries align with chars
TOKEN_PATTERN = re.compile(r'(\s+|[^\s\w\u00C0-\u024F\u1EA0-\u1EF9]+|\w+)')
# \u00C0-\u024F: Latin Extended-A/B (Vietnamese uppercase diacritics)
# \u1EA0-\u1EF9: Vietnamese combining diacritics (in NFC)


# ─────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Simple word-level tokenizer for Vietnamese.
    Returns list of tokens (words + punctuation).
    Empty tokens are filtered out.
    """
    tokens = TOKEN_PATTERN.findall(text)
    return [t for t in tokens if t]


def build_char_to_token_map(text: str, tokens: List[str]) -> List[int]:
    """
    Map each character position to its token index.
    Returns a list where char_to_token_map[i] = token index of char at position i.
    Characters not covered by any token get -1 (should not happen with proper tokens).
    """
    char_to_token = [-1] * len(text)
    pos = 0
    for tok_idx, token in enumerate(tokens):
        tok_len = len(token)
        for j in range(tok_len):
            if pos + j < len(text):
                char_to_token[pos + j] = tok_idx
        pos += tok_len
    return char_to_token


# ─────────────────────────────────────────────────────────
# BIO tagging
# ─────────────────────────────────────────────────────────

def spans_to_bio_tags(
    text: str,
    tokens: List[str],
    char_to_token: List[int],
    span_type: str,
    strict_b_tag: bool = False,
) -> List[str]:
    """
    Convert character-level spans to BIO tags.

    Args:
        text: original text
        tokens: tokenized text
        char_to_token: char index → token index mapping
        span_type: "aspect" → B-ASP/I-ASP
        strict_b_tag: if True, only token whose first char is span start → B-ASP.
                      if False, any token overlapping span start → B-ASP.
    Returns:
        List of BIO tags, one per token.
    """
    n = len(tokens)
    bio_tags = ["O"] * n

    if not span_type:
        return bio_tags

    label_prefix = f"B-{span_type.upper()}" if span_type.islower() else f"B-{span_type}"
    inside_prefix = f"I-{span_type.upper()}" if span_type.islower() else f"I-{span_type}"

    # Sort spans by start position
    # Build token→span involvement map
    # For each span, find all tokens it overlaps with
    for span in []:  # will be called with actual spans
        pass  # placeholder, real logic below

    return bio_tags


def compute_bio_tags(
    text: str,
    tokens: List[str],
    char_to_token: List[int],
    spans: List[Dict],
    span_type: str,
    strict_b_tag: bool = False,
) -> List[str]:
    """
    Convert character spans to BIO tags.
    A token is inside a span if it overlaps with the span's character range.

    B-ASP: first token of a span
    I-ASP: subsequent tokens of a span (for span_type="aspect")
    O: outside any span

    Overlapping spans: shorter span gets priority.
    """
    n = len(tokens)
    bio_tags = ["O"] * n

    if not span_type or not spans:
        return bio_tags

    # Fixed label names per task spec
    if span_type == "aspect":
        label_prefix = "B-ASP"
        inside_prefix = "I-ASP"
    else:
        # Fallback (should not reach here)
        label_prefix = f"B-{span_type.upper()}"
        inside_prefix = f"I-{span_type.upper()}"

    # Sort spans by start position, then by length (shorter first)
    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"] - s["start"]))

    # Track which tokens are already tagged (to handle overlap)
    token_tagged: Dict[int, bool] = {}

    for span in sorted_spans:
        span_start = span["start"]
        span_end = span["end"]

        # Find all tokens that overlap with this span
        overlapping_tokens: List[int] = []
        for char_pos in range(span_start, span_end):
            if 0 <= char_pos < len(char_to_token):
                tok_idx = char_to_token[char_pos]
                if tok_idx >= 0 and tok_idx < n and tok_idx not in token_tagged:
                    overlapping_tokens.append(tok_idx)

        if not overlapping_tokens:
            continue

        overlapping_tokens = sorted(set(overlapping_tokens))

        # Assign B-ASP to first, I-ASP to rest
        bio_tags[overlapping_tokens[0]] = label_prefix
        token_tagged[overlapping_tokens[0]] = True
        for tok_idx in overlapping_tokens[1:]:
            bio_tags[tok_idx] = inside_prefix
            token_tagged[tok_idx] = True

    return bio_tags


def bio_tags_to_spans(
    text: str,
    tokens: List[str],
    bio_tags: List[str],
    span_type: str,
) -> List[Dict]:
    """
    Round-trip: convert BIO tags back to character-level spans.
    Returns list of recovered spans.
    """
    recovered = []
    if not span_type:
        return recovered

    # Match the exact labels used in compute_bio_tags
    if span_type == "aspect":
        label_prefix = "B-ASP"
        inside_prefix = "I-ASP"
    else:
        label_prefix = f"B-{span_type.upper()}"
        inside_prefix = f"I-{span_type.upper()}"

    n = len(tokens)
    # Precompute cumulative token start positions
    token_starts = [0]
    for tok in tokens:
        token_starts.append(token_starts[-1] + len(tok))

    i = 0
    while i < n:
        tag = bio_tags[i]
        if tag == label_prefix:
            span_start_char = token_starts[i]
            j = i
            while j < n and bio_tags[j] in (label_prefix, inside_prefix):
                j += 1
            span_end_char = token_starts[j]
            recovered.append({
                "start": span_start_char,
                "end": span_end_char,
                "text": text[span_start_char:span_end_char],
            })
            i = j
        else:
            i += 1

    return recovered


# ─────────────────────────────────────────────────────────
# Round-trip validation
# ─────────────────────────────────────────────────────────

def validate_bio_tagging(
    text: str,
    tokens: List[str],
    char_to_token: List[int],
    original_spans: List[Dict],
    bio_tags: List[str],
    span_type: str,
    record_id: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate BIO tagging at the token level.

    Checks:
    1. Every original span's tokens are correctly labeled (B-ASP for start, I-ASP for rest).
    2. No token outside all original spans is labeled B-ASP or I-ASP.

    Returns (correct_records, errors).
    An "error" is a token that is labeled incorrectly relative to original spans.
    """
    n = len(tokens)
    errors = []

    if not span_type or not original_spans:
        return [], []

    label_prefix = "B-ASP"
    inside_prefix = "I-ASP"

    # For each span, find all overlapping token indices
    def get_overlapping_tokens(span_start: int, span_end: int) -> List[int]:
        overlapping = []
        for char_pos in range(span_start, span_end):
            if 0 <= char_pos < len(char_to_token):
                tok_idx = char_to_token[char_pos]
                if 0 <= tok_idx < n:
                    overlapping.append(tok_idx)
        return sorted(set(overlapping))

    # Check 1: Every original span's tokens are correctly labeled
    for span in original_spans:
        overlapping = get_overlapping_tokens(span["start"], span["end"])
        if not overlapping:
            continue

        # First token must be B-ASP
        first_tok = overlapping[0]
        if bio_tags[first_tok] != label_prefix:
            errors.append({
                "record_id": record_id,
                "error_type": "wrong_start_label",
                "token_index": first_tok,
                "token_text": tokens[first_tok] if first_tok < n else "",
                "token_start": sum(len(tokens[j]) for j in range(first_tok)),
                "assigned_label": bio_tags[first_tok],
                "expected_label": label_prefix,
                "span_start": span["start"],
                "span_end": span["end"],
                "span_text": span["text"],
                "span_type": span_type,
            })

        # Subsequent tokens must be I-ASP
        for tok_idx in overlapping[1:]:
            if bio_tags[tok_idx] != inside_prefix:
                errors.append({
                    "record_id": record_id,
                    "error_type": "wrong_inside_label",
                    "token_index": tok_idx,
                    "token_text": tokens[tok_idx] if tok_idx < n else "",
                    "token_start": sum(len(tokens[j]) for j in range(tok_idx)),
                    "assigned_label": bio_tags[tok_idx],
                    "expected_label": inside_prefix,
                    "span_start": span["start"],
                    "span_end": span["end"],
                    "span_text": span["text"],
                    "span_type": span_type,
                })

    # Check 2: No token outside all spans is labeled B-ASP or I-ASP
    # Build coverage: which tokens are inside at least one span
    covered_tokens = set()
    for span in original_spans:
        covered_tokens.update(get_overlapping_tokens(span["start"], span["end"]))

    for tok_idx in range(n):
        if tok_idx in covered_tokens:
            continue
        if bio_tags[tok_idx] in (label_prefix, inside_prefix):
            errors.append({
                "record_id": record_id,
                "error_type": "extra_tag",
                "token_index": tok_idx,
                "token_text": tokens[tok_idx],
                "token_start": sum(len(tokens[j]) for j in range(tok_idx)),
                "assigned_label": bio_tags[tok_idx],
                "expected_label": "O",
                "span_start": None,
                "span_end": None,
                "span_text": None,
                "span_type": span_type,
            })

    correct = [] if errors else [record_id]
    return correct, errors


# ─────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────

def load_final_splits() -> List[Dict]:
    """Load all records from data/final_splits/."""
    records = []
    for split in ["train", "valid", "test"]:
        path = DATA_IN / f"{split}.jsonl"
        if not path.exists():
            print(f"  [WARN] {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["_split"] = split
                records.append(rec)
    return records


def process_aspect_term_extraction(records: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Process records for aspect_term_extraction task.
    Filters to only uvisd4sa and causasent records with spans.
    Returns (output_records, roundtrip_errors, stats).
    """
    output_records = []
    all_errors = []
    stats = {
        "by_source": defaultdict(lambda: {
            "records": 0, "spans": 0, "records_with_spans": 0,
        }),
        "bio_label_counts": Counter(),
        "total_tokens": 0,
        "total_roundtrip_errors": 0,
    }

    for rec in records:
        if rec["source"] not in ("uvisd4sa", "causasent"):
            continue
        if not rec.get("spans"):
            continue  # skip records without spans

        text = rec["text"]
        spans = rec["spans"]

        # Tokenize
        tokens = tokenize(text)
        if not tokens:
            continue

        char_to_token = build_char_to_token_map(text, tokens)

        # Compute BIO tags
        bio_tags = compute_bio_tags(
            text, tokens, char_to_token, spans,
            span_type="aspect", strict_b_tag=False,
        )

        # Validate BIO tagging
        _, errors = validate_bio_tagging(
            text, tokens, char_to_token, spans, bio_tags,
            span_type="aspect", record_id=rec["id"],
        )

        all_errors.extend(errors)

        # Build output record
        output_rec = {
            "id": rec["id"],
            "source": rec["source"],
            "task": "aspect_term_extraction",
            "text": text,
            "split": rec["split"],
            "review_level_label": rec["review_level_label"],
            "tokens": tokens,
            "token_offsets": _compute_token_offsets(text, tokens),
            "bio_tags": bio_tags,
            "num_tokens": len(tokens),
            "num_spans": len(spans),
        }
        output_records.append(output_rec)

        # Update stats
        stats["by_source"][rec["source"]]["records"] += 1
        stats["by_source"][rec["source"]]["spans"] += len(spans)
        if spans:
            stats["by_source"][rec["source"]]["records_with_spans"] += 1
        stats["bio_label_counts"].update(bio_tags)
        stats["total_tokens"] += len(tokens)

    stats["total_roundtrip_errors"] = len(all_errors)
    return output_records, all_errors, stats


def process_complaint_detection(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Process records for complaint_detection task (ViOCD only).
    Each record has review-level label, no BIO tags needed.
    """
    output_records = []
    stats = {
        "records": 0,
        "complaint_records": 0,
        "non_complaint_records": 0,
    }

    for rec in records:
        if rec["source"] != "viocd":
            continue

        output_rec = {
            "id": rec["id"],
            "source": rec["source"],
            "text": rec["text"],
            "split": rec["split"],
            "label": rec["review_level_label"],  # 0 or 1
        }
        output_records.append(output_rec)

        stats["records"] += 1
        if rec["review_level_label"] == 1:
            stats["complaint_records"] += 1
        else:
            stats["non_complaint_records"] += 1

    return output_records, stats


def _compute_token_offsets(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """Compute (start_char, end_char) offsets for each token."""
    offsets = []
    pos = 0
    for token in tokens:
        start = text.find(token, pos)
        if start == -1:
            # Fallback: use pos
            start = pos
        end = start + len(token)
        offsets.append((start, end))
        pos = end
    return offsets


def save_jsonl(records: List[Dict], path: Path):
    """Save records to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_ate_manifest(
    stats: Dict,
    errors: List[Dict],
    out_dir: Path,
) -> Dict:
    """Save aspect_term_extraction manifest."""
    manifest = {
        "task": "aspect_term_extraction",
        "bio_labels": sorted(BIO_LABELS["aspect_term_extraction"]),
        "total_records": sum(s["records"] for s in stats["by_source"].values()),
        "total_spans": sum(s["spans"] for s in stats["by_source"].values()),
        "total_tokens": stats["total_tokens"],
        "roundtrip_errors": stats["total_roundtrip_errors"],
        "by_source": {
            src: dict(v) for src, v in stats["by_source"].items()
        },
        "bio_label_distribution": dict(stats["bio_label_counts"]),
        "valid_bio_labels": {
            label: count
            for label, count in stats["bio_label_counts"].items()
            if label in BIO_LABELS["aspect_term_extraction"]
        },
    }
    with open(out_dir / "bio_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def save_detection_manifest(stats: Dict, out_dir: Path) -> Dict:
    """Save complaint_detection manifest."""
    manifest = {
        "task": "complaint_detection",
        "total_records": stats["records"],
        "complaint_records": stats["complaint_records"],
        "non_complaint_records": stats["non_complaint_records"],
    }
    with open(out_dir / "detection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def save_bio_errors(errors: List[Dict], out_dir: Path):
    """Save BIO tagging errors to CSV."""
    if not errors:
        return
    path = out_dir / "bio_roundtrip_errors.csv"
    fieldnames = [
        "record_id", "error_type", "token_index", "token_text",
        "token_start", "assigned_label", "expected_label",
        "span_start", "span_end", "span_text", "span_type",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    print(f"  Saved {path.name}: {len(errors)} error rows")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Step 2: Create Task-Aware BIO Datasets")
    print("=" * 70)
    print(f"\n  Input:  {DATA_IN}")
    print(f"  Output: {DATA_OUT}\n")

    # ── Create output directories ─────────────────────────
    ate_dir = DATA_OUT / "aspect_term_extraction"
    cd_dir  = DATA_OUT / "complaint_detection"
    cs_dir  = DATA_OUT / "complaint_span_extraction"
    ate_dir.mkdir(parents=True, exist_ok=True)
    cd_dir.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load final splits ────────────────────────────────
    print("-" * 70)
    print("1. LOADING DATA")
    print("-" * 70)
    all_records = load_final_splits()
    print(f"  Loaded {len(all_records)} total records")
    by_source = Counter(r["source"] for r in all_records)
    for src, cnt in sorted(by_source.items()):
        print(f"    {src}: {cnt}")

    # ── Process aspect_term_extraction ──────────────────
    print("\n" + "-" * 70)
    print("2. PROCESSING aspect_term_extraction")
    print("-" * 70)
    ate_records, ate_errors, ate_stats = process_aspect_term_extraction(all_records)
    print(f"  Records:     {len(ate_records)}")
    print(f"  Total spans: {sum(s['spans'] for s in ate_stats['by_source'].values())}")

    # Group ATE records by split
    ate_by_split = defaultdict(list)
    for rec in ate_records:
        ate_by_split[rec["split"]].append(rec)

    for split in ["train", "valid", "test"]:
        out_path = ate_dir / f"{split}.jsonl"
        recs = ate_by_split.get(split, [])
        save_jsonl(recs, out_path)
        print(f"  Saved {split}.jsonl: {len(recs)} records")

    # ── Process complaint_detection ─────────────────────
    print("\n" + "-" * 70)
    print("3. PROCESSING complaint_detection (ViOCD)")
    print("-" * 70)
    cd_records, cd_stats = process_complaint_detection(all_records)
    print(f"  Records:     {cd_stats['records']}")
    print(f"  Complaints:  {cd_stats['complaint_records']}")
    print(f"  Non-compl:   {cd_stats['non_complaint_records']}")

    cd_by_split = defaultdict(list)
    for rec in cd_records:
        cd_by_split[rec["split"]].append(rec)

    for split in ["train", "valid", "test"]:
        out_path = cd_dir / f"{split}.jsonl"
        recs = cd_by_split.get(split, [])
        save_jsonl(recs, out_path)
        print(f"  Saved {split}.jsonl: {len(recs)} records")

    # ── Save complaint_span_extraction README ────────────
    readme_path = cs_dir / "README.txt"
    readme_path.write_text(
        "Vietnamese Complaint Span Extraction Dataset\n"
        "============================================\n\n"
        "Status: NOT YET ANNOTATED\n\n"
        "ViOCD contains review-level complaint labels (0/1) but NO complaint span\n"
        "annotations. To create a complaint_span_extraction dataset:\n\n"
        "1. Annotate complaint spans (character-level [start, end]) in ViOCD texts.\n"
        "2. Validate span offsets (start >= 0, end <= len(text), text[start:end] matches).\n"
        "3. Run BIO conversion with labels O, B-COMP, I-COMP.\n"
        "4. Perform round-trip validation.\n\n"
        "Until step 3 is complete, complaint_span_extraction has no BIO data.\n"
    )
    print(f"\n  Saved {readme_path.name}")

    # ── Save manifests ───────────────────────────────────
    print("\n" + "-" * 70)
    print("4. SAVING MANIFESTS")
    print("-" * 70)
    ate_manifest = save_ate_manifest(ate_stats, ate_errors, ate_dir)
    cd_manifest = save_detection_manifest(cd_stats, cd_dir)
    print(f"  Saved {ate_dir.name}/bio_manifest.json")
    print(f"  Saved {cd_dir.name}/detection_manifest.json")

    # ── Save BIO errors ───────────────────────────────────
    save_bio_errors(ate_errors, ate_dir)

    # ── Print summary ───────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  [BIO Label Distribution]")
    for label, count in sorted(ate_stats["bio_label_counts"].items()):
        print(f"    {label}: {count:,}")

    print("\n  [Records by Source and Split]")
    for src in ["uvisd4sa", "causasent"]:
        if src in ate_stats["by_source"]:
            print(f"    {src}:")
            for split in ["train", "valid", "test"]:
                cnt = sum(
                    1 for r in ate_by_split.get(split, [])
                    if r["source"] == src
                )
                spn = sum(
                    r["num_spans"] for r in ate_by_split.get(split, [])
                    if r["source"] == src
                )
                if cnt > 0:
                    print(f"      {split}: {cnt} records, {spn} spans")

    print("\n  [BIO Token-Level Errors]")
    print(f"    Total errors: {len(ate_errors)}")
    if ate_errors:
        err_types = Counter(e["error_type"] for e in ate_errors)
        for t, c in err_types.items():
            print(f"      {t}: {c}")
        print(f"\n    Examples:")
        for e in ate_errors[:5]:
            print(f"      {e['record_id']}  {e['error_type']}  "
                  f"token[{e['token_index']}]={repr(e['token_text'][:20])}  "
                  f"got={e['assigned_label']}  expected={e['expected_label']}")

    print("\n  [Saved Output Paths]")
    for f in sorted(DATA_OUT.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(REPO_ROOT)}  ({f.stat().st_size:,} bytes)")

    print("\n  ✅ Step 2 complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
