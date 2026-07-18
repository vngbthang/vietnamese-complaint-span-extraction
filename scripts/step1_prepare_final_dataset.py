#!/usr/bin/env python3
"""
Step 1: Prepare Final Dataset
=============================
Normalizes three Vietnamese datasets into a unified format.

Dataset roles:
  - ViOCD:          review-level complaint detection only
  - ViOCD-Span:     complaint span extraction (to be annotated/validated next)
  - UIT-ViSD4SA:    auxiliary aspect/sentiment spans (NOT complaint spans)
  - CausaSent-ATE:  auxiliary aspect term extraction (NOT complaint spans)

Decision log:
  - 41 invalid UIT-ViSD4SA spans: keep parent records, remove only invalid spans
  - Do NOT merge datasets into one complaint span dataset
  - Preserve source and span_type in all records
  - B-COMP/I-COMP only for complaint spans
  - B-ASP/I-ASP only for aspect spans
"""

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_OUT = REPO_ROOT / "data" / "final_splits"
DATA_OUT.mkdir(parents=True, exist_ok=True)

SPLIT_MAP = {
    "train": "train",
    "dev":   "valid",
    "valid": "valid",
    "validation": "valid",
    "test": "test",
}

# ─────────────────────────────────────────────────────────
# Per-dataset configuration
# Each dataset has: source tag, task type, and column mapping
# ─────────────────────────────────────────────────────────
DS_CONFIG: Dict[str, Dict[str, Any]] = {
    "viocd": {
        "path":              DATA_RAW / "viocd",
        "files": {
            "train": "train.jsonl",
            "valid": "validation.jsonl",
            "test":  "test.jsonl",
        },
        "source":           "viocd",
        "task":             "complaint_detection",
        "id_col":           "id",
        "text_col":         "text",
        "span_col":         None,          # no spans in raw data
        "review_label_col": "label",       # 1.0 = complaint, 0.0 = no complaint
        "offset_type":      "char",
        "span_type":        None,          # no spans
    },
    "uvisd4sa": {
        "path":              DATA_RAW / "uvisd4sa",
        "files": {
            "train": "train.jsonl",
            "dev":   "dev.jsonl",
            "test":  "test.jsonl",
        },
        "source":           "uvisd4sa",
        "task":             "aspect_term_extraction",
        "id_col":           None,           # no ID → generate
        "text_col":         "text",
        "span_col":         "labels",       # [[start, end, ASPECT#POLARITY], ...]
        "review_label_col": None,           # derived from NEGATIVE polarity
        "offset_type":      "byte",         # MUST convert to char
        "span_type":        "aspect",
    },
    "causasent": {
        "path":              DATA_RAW / "causasent_ate_v2",
        "files": {
            "train": "train.jsonl",
            "valid": "validation.jsonl",
            "test":  "test.jsonl",
        },
        "source":           "causasent",
        "task":             "aspect_term_extraction",
        "id_col":           "id",
        "text_col":         "review",
        "span_col":         "annotations",   # [{aspect_term, aspect_term_span, ...}, ...]
        "review_label_col": None,           # derived from negative sentiment
        "offset_type":      "char",
        "span_type":        "aspect",
    },
}


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def safe_parse_span_col(value: Any) -> List:
    """Parse the span column which may be a list, string, or None."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value or value in ("[]", "None"):
            return []
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def byte_to_char_offset(text: str, byte_start: int, byte_end: int) -> Tuple[int, int]:
    """Convert UTF-8 byte offsets → Python character offsets."""
    text_bytes = text.encode("utf-8")
    try:
        char_start = len(text_bytes[:byte_start].decode("utf-8"))
        char_end   = len(text_bytes[:byte_end].decode("utf-8"))
        return char_start, char_end
    except UnicodeDecodeError:
        return byte_start, byte_end


def derive_review_label(raw_label: Any, spans: List[Dict], ds_name: str) -> Optional[int]:
    """
    Derive or extract the review-level complaint label.

    For ViOCD: use the raw label directly (1 = complaint, 0 = no complaint).
    For others: derive from spans (negative polarity → 1, else 0).
    Returns None if no label can be determined.
    """
    # ViOCD: explicit review-level label
    if ds_name == "viocd":
        if raw_label is not None:
            return int(float(raw_label))
        return None

    # UIT-ViSD4SA: negative polarity spans → complaint
    if ds_name == "uvisd4sa":
        for span in spans:
            tag = span.get("_raw_tag", "")
            if "#" in tag:
                polarity = tag.split("#")[-1].upper()
                if polarity == "NEGATIVE":
                    return 1
        return 0

    # CausaSent: negative sentiment → complaint
    if ds_name == "causasent":
        for span in spans:
            sentiment = span.get("sentiment", "").lower()
            if sentiment == "negative":
                return 1
        return 0

    return None


def parse_spans(raw_spans: List, dataset: str, text: str) -> List[Dict]:
    """
    Parse raw spans into unified {start, end, text, span_type} format.
    Applies byte→char conversion for UIT-ViSD4SA.
    """
    parsed = []
    span_type = DS_CONFIG[dataset]["span_type"]

    for item in raw_spans:
        if not item:
            continue

        if dataset == "uvisd4sa":
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            byte_start, byte_end, tag = int(item[0]), int(item[1]), str(item[2])
            char_start, char_end = byte_to_char_offset(text, byte_start, byte_end)

        elif dataset == "causasent":
            if not isinstance(item, dict):
                continue
            span_coords = item.get("aspect_term_span", [])
            if not isinstance(span_coords, (list, tuple)) or len(span_coords) != 2:
                continue
            char_start, char_end = int(span_coords[0]), int(span_coords[1])
            tag = item.get("aspect_category", "")

        elif dataset == "viocd":
            continue  # no spans

        else:
            continue

        parsed_entry: Dict[str, Any] = {
            "start":     char_start,
            "end":       char_end,
            "text":      text[char_start:char_end] if (0 <= char_start < char_end <= len(text)) else "",
            "span_type": span_type,
            "_raw_tag":  tag,
        }
        if dataset == "causasent":
            parsed_entry["sentiment"] = item.get("sentiment", "")
        parsed.append(parsed_entry)

    return parsed


def validate_span(span: Dict, text: str) -> Tuple[bool, str]:
    """
    Validate a single span: bounds and text match.
    Returns (is_valid, error_reason).
    """
    start = span["start"]
    end   = span["end"]

    if start < 0:
        return False, f"start={start} < 0"
    if end > len(text):
        return False, f"end={end} > len(text)={len(text)}"
    if start >= end:
        return False, f"start={start} >= end={end}"
    if start >= len(text):
        return False, f"start={start} >= len(text)={len(text)}"

    actual = text[start:end]
    if actual != span.get("text", ""):
        return False, f"text mismatch: expected {repr(actual)}, got {repr(span.get('text',''))}"

    return True, ""


def check_duplicates(records: List[Dict]) -> Dict:
    """Check for duplicate IDs across and within splits."""
    by_id: Dict[str, List[str]] = defaultdict(list)
    by_split_and_source: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rec in records:
        by_id[rec["id"]].append(rec["split"])
        key = (rec["split"], rec["source"])
        by_split_and_source[key][rec["id"]] += 1

    duplicate_ids = {uid: splits for uid, splits in by_id.items() if len(splits) > 1}

    # Within-split duplicates: same (split, source, id) appears more than once
    within_split_dups: Dict[str, List[str]] = {}
    for (split_, source_), id_counts in by_split_and_source.items():
        dups = [uid for uid, cnt in id_counts.items() if cnt > 1]
        if dups:
            key = f"{split_}_{source_}"
            within_split_dups[key] = dups

    return {
        "duplicate_ids":         duplicate_ids,
        "within_split_duplicates": within_split_dups,
        "total_unique_ids":      len(by_id),
    }


# ─────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────

def process_dataset(ds_name: str, cfg: Dict) -> Dict[str, List[Dict]]:
    """Process a single dataset, returning records grouped by split."""
    split_records: Dict[str, List[Dict]] = defaultdict(list)
    id_counter: Dict[str, int] = defaultdict(int)

    for orig_split, filename in cfg["files"].items():
        filepath = cfg["path"] / filename
        if not filepath.exists():
            print(f"  [WARN] File not found: {filepath}")
            continue

        normalized_split = SPLIT_MAP.get(orig_split, orig_split)
        print(f"  [{ds_name}] {filename} → '{normalized_split}'")

        with open(filepath, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # ── Text ─────────────────────────────────────
                text = raw.get(cfg["text_col"], "").strip()
                if not text:
                    continue

                # ── ID ───────────────────────────────────────
                raw_id = raw.get(cfg["id_col"])
                if raw_id is None:
                    id_counter[normalized_split] += 1
                    unique_id = f"{ds_name}_{normalized_split}_{id_counter[normalized_split]}"
                else:
                    unique_id = f"{ds_name}_{raw_id}"

                # ── Parse spans ──────────────────────────────
                raw_spans = safe_parse_span_col(raw.get(cfg["span_col"]))
                spans = parse_spans(raw_spans, ds_name, text)

                # ── Review-level label ──────────────────────
                raw_label = raw.get(cfg["review_label_col"])
                review_label = derive_review_label(raw_label, spans, ds_name)

                record = {
                    "id":                  unique_id,
                    "source":              cfg["source"],
                    "task":                cfg["task"],
                    "text":                text,
                    "split":               normalized_split,
                    "review_level_label":  review_label,
                    "spans":               spans,
                }
                split_records[normalized_split].append(record)

    return split_records


def validate_and_filter_spans(
    records: List[Dict]
) -> Tuple[List[Dict], List[Dict], int]:
    """
    Validate every span. Invalid spans are removed but parent records are kept.
    Returns (records, invalid_span_log, invalid_count).
    """
    valid_records = []
    invalid_log = []
    total_invalid = 0

    for rec in records:
        text = rec["text"]
        valid_spans = []
        rec_invalid = []

        for span in rec.get("spans", []):
            is_valid, reason = validate_span(span, text)
            if is_valid:
                valid_spans.append(span)
            else:
                total_invalid += 1
                rec_invalid.append({**span, "_reason": reason})
                invalid_log.append({
                    "record_id":     rec["id"],
                    "source":        rec["source"],
                    "task":          rec["task"],
                    "split":         rec["split"],
                    "start":         span["start"],
                    "end":           span["end"],
                    "text_extracted": span.get("text", ""),
                    "error":         reason,
                })

        rec["spans"] = valid_spans
        valid_records.append(rec)

    return valid_records, invalid_log, total_invalid


def _init_stats_keys() -> Dict[str, int]:
    """Return a dict with all expected stat keys initialized to 0."""
    return {
        "records": 0,
        "spans": 0,
        "records_with_spans": 0,
        "records_without_spans": 0,
        "complaint_records": 0,
        "non_complaint_records": 0,
    }


def compute_stats(records: List[Dict]) -> Dict:
    """Compute per-split, per-source statistics."""
    splits: Dict[str, Dict[str, int]] = defaultdict(_init_stats_keys)
    sources: Dict[str, Dict[str, int]] = defaultdict(_init_stats_keys)
    span_types: Dict[str, int] = defaultdict(int)

    for rec in records:
        s = rec["split"]
        src = rec["source"]
        splits[s]["records"] += 1
        sources[src]["records"] += 1

        n_spans = len(rec["spans"])
        splits[s]["spans"] += n_spans
        sources[src]["spans"] += n_spans
        span_types["all"] += n_spans

        if n_spans > 0:
            splits[s]["records_with_spans"] += 1
            sources[src]["records_with_spans"] += 1
        else:
            splits[s]["records_without_spans"] += 1
            sources[src]["records_without_spans"] += 1

        lbl = rec["review_level_label"]
        if lbl is not None:
            if lbl == 1:
                splits[s]["complaint_records"] += 1
                sources[src]["complaint_records"] += 1
            else:
                splits[s]["non_complaint_records"] += 1
                sources[src]["non_complaint_records"] += 1

        for span in rec["spans"]:
            span_types[span.get("span_type", "unknown")] += 1

    result = {
        "by_split":    dict(splits),
        "by_source":   dict(sources),
        "by_span_type": dict(span_types),
    }
    return result


def save_splits(records: List[Dict], out_dir: Path):
    """Write train.jsonl, valid.jsonl, test.jsonl."""
    by_split: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        by_split[rec["split"]].append(rec)

    for split_name in ["train", "valid", "test"]:
        out_path = out_dir / f"{split_name}.jsonl"
        recs = by_split.get(split_name, [])
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in recs:
                # Drop internal _fields if any sneak through
                out = {k: v for k, v in rec.items() if not k.startswith("_")}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  {split_name}.jsonl: {len(recs)} records")


def save_invalid_spans_csv(invalid_log: List[Dict], out_dir: Path):
    """Save invalid spans to CSV for reproducibility."""
    if not invalid_log:
        return
    path = out_dir / "invalid_spans.csv"
    fieldnames = ["record_id", "source", "task", "split", "start", "end", "text_extracted", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(invalid_log)
    print(f"  invalid_spans.csv: {len(invalid_log)} rows")


def save_manifest(
    records: List[Dict],
    invalid_count: int,
    dup_check: Dict,
    out_dir: Path,
) -> Dict:
    """Build and save dataset_manifest.json."""
    stats = compute_stats(records)
    total_spans = sum(len(r["spans"]) for r in records)

    manifest = {
        "total_records":   len(records),
        "total_spans":     total_spans,
        "invalid_spans_removed": invalid_count,
        "duplicate_id_count": len(dup_check["duplicate_ids"]),
        "splits": {
            split_name: {
                "records":                 stats["by_split"][split_name].get("records", 0),
                "spans":                   stats["by_split"][split_name].get("spans", 0),
                "records_with_spans":      stats["by_split"][split_name].get("records_with_spans", 0),
                "records_without_spans":    stats["by_split"][split_name].get("records_without_spans", 0),
                "complaint_records":        stats["by_split"][split_name].get("complaint_records", 0),
                "non_complaint_records":    stats["by_split"][split_name].get("non_complaint_records", 0),
            }
            for split_name in ["train", "valid", "test"]
            if split_name in stats["by_split"]
        },
        "by_source": {
            src: {
                "records":               stats["by_source"][src].get("records", 0),
                "spans":                 stats["by_source"][src].get("spans", 0),
                "records_with_spans":    stats["by_source"][src].get("records_with_spans", 0),
                "records_without_spans": stats["by_source"][src].get("records_without_spans", 0),
                "complaint_records":     stats["by_source"][src].get("complaint_records", 0),
                "non_complaint_records": stats["by_source"][src].get("non_complaint_records", 0),
                "task":                 DS_CONFIG[src]["task"],
            }
            for src in DS_CONFIG
            if src in stats["by_source"]
        },
        "by_span_type": stats["by_span_type"],
        "decisions": {
            "invalid_spans":        "keep_parent_remove_spans",
            "invalid_spans_removed": invalid_count,
            "merge_all_datasets":   False,
            "bio_scheme": {
                "complaint_spans": "O, B-COMP, I-COMP",
                "aspect_spans":    "O, B-ASP, I-ASP",
            },
        },
    }

    manifest_path = out_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Step 1: Prepare Final Dataset")
    print("=" * 70)
    print(f"\n  Input:  {DATA_RAW}")
    print(f"  Output: {DATA_OUT}\n")

    # ── 1. Column mapping ──────────────────────────────────
    print("-" * 70)
    print("1. DETECTED COLUMN MAPPING")
    print("-" * 70)
    for ds_name, cfg in DS_CONFIG.items():
        sample_file = next(iter(cfg["files"].values()))
        with open(cfg["path"] / sample_file, encoding="utf-8") as f:
            raw = json.loads(f.readline())
        print(f"\n  [{ds_name}]  source={cfg['source']}  task={cfg['task']}")
        print(f"    raw columns  : {list(raw.keys())}")
        print(f"    text_col     : {cfg['text_col']!r}")
        print(f"    id_col       : {cfg['id_col']!r}")
        print(f"    span_col     : {cfg['span_col']!r}")
        print(f"    label_col    : {cfg['review_label_col']!r}")
        print(f"    offset_type  : {cfg['offset_type']}")
        print(f"    span_type    : {cfg['span_type']!r}")

    # ── 2. Process datasets ─────────────────────────────────
    print("\n" + "-" * 70)
    print("2. PROCESSING DATASETS")
    print("-" * 70)
    all_split_records: Dict[str, List[Dict]] = defaultdict(list)
    for ds_name, cfg in DS_CONFIG.items():
        results = process_dataset(ds_name, cfg)
        for split_name, recs in results.items():
            all_split_records[split_name].extend(recs)
            print(f"  [{ds_name}] → {split_name}: {len(recs)} records")

    # ── 3. Combine ─────────────────────────────────────────
    all_records = []
    for split_name in ["train", "valid", "test"]:
        all_records.extend(all_split_records[split_name])
    print(f"\n  Total records: {len(all_records)}")

    # ── 4. Validate spans ───────────────────────────────────
    print("\n" + "-" * 70)
    print("3. VALIDATING SPANS")
    print("-" * 70)
    valid_records, invalid_log, total_invalid = validate_and_filter_spans(all_records)
    print(f"  Total records:         {len(all_records)}")
    print(f"  Valid records kept:    {len(valid_records)}")
    print(f"  Invalid spans removed:  {total_invalid}")

    if invalid_log:
        print("\n  Examples of invalid spans:")
        seen = set()
        for item in invalid_log:
            key = (item["record_id"], item["start"], item["end"])
            if key in seen:
                continue
            seen.add(key)
            print(f"    {item['record_id']}  start={item['start']}  end={item['end']}"
                  f"  extracted={repr(item['text_extracted'][:25])}  → {item['error']}")

    # ── 5. Check duplicates ─────────────────────────────────
    print("\n" + "-" * 70)
    print("4. CHECKING DUPLICATE IDs")
    print("-" * 70)
    dup_check = check_duplicates(valid_records)
    print(f"  Total unique IDs:     {dup_check['total_unique_ids']}")
    print(f"  Duplicate IDs found:   {len(dup_check['duplicate_ids'])}")
    if dup_check["duplicate_ids"]:
        for uid, splits in list(dup_check["duplicate_ids"].items())[:5]:
            print(f"    {uid} → splits: {splits}")
    else:
        print("  No duplicate IDs found.")

    # ── 6. Stats ───────────────────────────────────────────
    print("\n" + "-" * 70)
    print("5. SPLIT + SOURCE DISTRIBUTION")
    print("-" * 70)
    stats = compute_stats(valid_records)

    for split_name in ["train", "valid", "test"]:
        if split_name not in stats["by_split"]:
            continue
        s = stats["by_split"][split_name]
        print(f"\n  [{split_name}]")
        print(f"    records:                    {s['records']:>6}")
        print(f"    spans:                      {s['spans']:>6}")
        print(f"    records with spans:         {s['records_with_spans']:>6}")
        print(f"    records without spans:      {s['records_without_spans']:>6}")
        print(f"    complaint records:          {s['complaint_records']:>6}")
        print(f"    non-complaint records:      {s['non_complaint_records']:>6}")

    print(f"\n  [by source]")
    for src in DS_CONFIG:
        if src not in stats["by_source"]:
            continue
        s = stats["by_source"][src]
        task = DS_CONFIG[src]["task"]
        print(f"    {src} ({task}):")
        print(f"      records={s['records']}  spans={s['spans']}"
              f"  complaint={s.get('complaint_records', 0)}"
              f"  non-complaint={s.get('non_complaint_records', 0)}")

    print(f"\n  [by span_type]")
    for stype, cnt in stats["by_span_type"].items():
        print(f"    {stype}: {cnt}")

    # ── 7. Save outputs ────────────────────────────────────
    print("\n" + "-" * 70)
    print("6. SAVING OUTPUTS")
    print("-" * 70)
    save_splits(valid_records, DATA_OUT)
    save_invalid_spans_csv(invalid_log, DATA_OUT)
    manifest = save_manifest(valid_records, total_invalid, dup_check, DATA_OUT)

    # ── 8. Final summary ───────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total records:              {manifest['total_records']:>6}")
    print(f"  Total spans:                {manifest['total_spans']:>6}")
    print(f"  Invalid spans removed:      {manifest['invalid_spans_removed']:>6}")
    print(f"  Duplicate IDs:               {manifest['duplicate_id_count']:>6}")

    print(f"\n  Output: {DATA_OUT}/")
    for f in sorted(DATA_OUT.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print("\n  ✅ Step 1 complete. Safe to proceed to BIO conversion (Step 2).")
    sys.exit(0)


if __name__ == "__main__":
    main()
