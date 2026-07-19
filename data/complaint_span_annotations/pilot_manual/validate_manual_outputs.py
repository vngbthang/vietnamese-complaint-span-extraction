#!/usr/bin/env python3
"""
Step 3A-Manual: Validator for Pasted LLM Batch Outputs
=====================================================
Loads LLM JSON outputs from batch_*_output.json files, validates spans,
and produces the final annotated dataset.

Usage:
  python3 validate_manual_outputs.py

Expected input files (in same directory as this script):
  ../pilot/pilot_candidates.jsonl
  batch_outputs/batch_001_output.json
  batch_outputs/batch_002_output.json
  batch_outputs/batch_003_output.json
  batch_outputs/batch_004_output.json

Output files:
  pilot_manual_validated.jsonl
  pilot_manual_invalid.csv
  pilot_manual_manifest.json
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
CANDIDATES_PATH = REPO_ROOT / "data" / "complaint_span_annotations" / "pilot" / "pilot_candidates.jsonl"
BATCH_OUT_DIR   = SCRIPT_DIR / "batch_outputs"
OUTPUT_DIR      = SCRIPT_DIR


# ─────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────

def find_span_offset(text: str, span_text: str) -> Optional[Tuple[int, int]]:
    """Find character offset of span_text in text via exact string matching."""
    if not span_text or not isinstance(span_text, str):
        return None
    span_text = span_text.strip()
    if not span_text:
        return None
    idx = text.find(span_text)
    if idx >= 0:
        return (idx, idx + len(span_text))
    return None


def is_invalid_span_text(text: str) -> bool:
    """Return True if span text is empty/whitespace/punctuation-only."""
    if not text or not isinstance(text, str):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    import string
    punct = set(string.punctuation)
    all_punct_or_space = all(c in punct | {" ", "\t", "\n", "\r", "\u00a0"} for c in stripped)
    return all_punct_or_space


def load_batch_outputs() -> List[Dict]:
    """Load all batch_*_output.json files."""
    results = []
    batch_files = sorted(BATCH_OUT_DIR.glob("batch_*_output.json"))
    for bf in batch_files:
        with open(bf, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            results.extend(data)
        else:
            print(f"  [WARN] {bf.name} root is not a list, skipping")
    return results


def parse_record_output(
    record: Dict,
    llm_entry: Optional[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse and validate a single LLM output entry.
    Returns (validated_spans, invalid_entries).
    """
    rid = record["id"]
    text = record["text"]
    validated = []
    invalid = []

    if llm_entry is None:
        invalid.append({
            "record_id": rid,
            "split": record["split"],
            "reason": "missing_output",
            "detail": "",
            "text": text,
        })
        return [], invalid

    if not isinstance(llm_entry, dict):
        invalid.append({
            "record_id": rid,
            "split": record["split"],
            "reason": "not_dict",
            "detail": str(type(llm_entry)),
            "text": text,
        })
        return [], invalid

    spans = llm_entry.get("complaint_spans", [])
    if not isinstance(spans, list):
        invalid.append({
            "record_id": rid,
            "split": record["split"],
            "reason": "complaint_spans_not_list",
            "detail": str(type(spans)),
            "text": text,
        })
        return [], invalid

    for idx, span in enumerate(spans):
        if not isinstance(span, dict):
            invalid.append({
                "record_id": rid,
                "split": record["split"],
                "reason": "span_not_dict",
                "span_index": idx,
                "detail": str(type(span)),
                "text": text,
            })
            continue

        span_text = span.get("text", "")

        if is_invalid_span_text(span_text):
            invalid.append({
                "record_id": rid,
                "split": record["split"],
                "reason": "invalid_span_text",
                "span_index": idx,
                "detail": repr(span_text[:50]),
                "text": text,
            })
            continue

        offset = find_span_offset(text, span_text)
        if offset is None:
            invalid.append({
                "record_id": rid,
                "split": record["split"],
                "reason": "text_not_in_review",
                "span_index": idx,
                "detail": repr(span_text[:100]),
                "text": text,
            })
            continue

        validated.append({
            "text":           span_text,
            "start":          offset[0],
            "end":            offset[1],
            "span_type":      "complaint",
            "annotation_mode": "manual_llm_web",
            "offset_source":   "exact_string_match",
        })

    return validated, invalid


def remove_exact_duplicates(spans: List[Dict]) -> List[Dict]:
    """Remove duplicate spans (same text)."""
    seen = set()
    unique = []
    for s in spans:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    return unique


def resolve_nested_spans(spans: List[Dict]) -> List[Dict]:
    """Remove spans fully contained within other spans (keep the longer one)."""
    if len(spans) <= 1:
        return spans
    sorted_spans = sorted(spans, key=lambda s: (s["end"] - s["start"], s["start"]))
    resolved = []
    for span in sorted_spans:
        s_start, s_end = span["start"], span["end"]
        dominated = any(
            e_start <= s_start and s_end <= e_end
            for e in resolved
            for e_start, e_end in [(e["start"], e["end"])]
        )
        if dominated:
            continue
        resolved = [e for e in resolved
                    if not (s_start <= e["start"] and e["end"] <= s_end)]
        resolved.append(span)
    return resolved


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Step 3A-Manual: Validate Pasted LLM Batch Outputs")
    print("=" * 70)
    print(f"\n  Candidates: {CANDIDATES_PATH}")
    print(f"  Batch dir:  {BATCH_OUT_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}\n")

    # ── 1. Load candidates ────────────────────────────────
    print("-" * 70)
    print("1. LOADING PILOT CANDIDATES")
    print("-" * 70)
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f]
    print(f"  Loaded {len(candidates)} candidates")
    rec_by_id = {r["id"]: r for r in candidates}

    # ── 2. Load batch outputs ─────────────────────────────
    print("\n" + "-" * 70)
    print("2. LOADING LLM BATCH OUTPUTS")
    print("-" * 70)
    llm_results = load_batch_outputs()
    print(f"  Loaded {len(llm_results)} LLM output entries")

    if not llm_results:
        print("  [WARN] No batch_*_output.json files found in batch_outputs/")
        print("  [WARN] Please save your LLM outputs before running this validator.")
        sys.exit(1)

    llm_by_id = {entry.get("id"): entry for entry in llm_results}

    # ── 3. Validate all records ──────────────────────────
    print("\n" + "-" * 70)
    print("3. VALIDATING SPANS")
    print("-" * 70)
    validated_records = []
    all_invalid = []
    error_counts = Counter()

    for rec in candidates:
        rid = rec["id"]
        llm_entry = llm_by_id.get(rid)
        spans, invalid = parse_record_output(rec, llm_entry)
        error_counts["missing_output"] += 1 if llm_entry is None else 0
        error_counts["invalid_span_text"] += sum(
            1 for e in invalid if e["reason"] == "invalid_span_text"
        )
        error_counts["text_not_in_review"] += sum(
            1 for e in invalid if e["reason"] == "text_not_in_review"
        )
        error_counts["json_parse_error"] += sum(
            1 for e in invalid if e["reason"] in ("not_dict", "complaint_spans_not_list", "span_not_dict")
        )

        for entry in invalid:
            entry["batch"] = ""  # placeholder
            all_invalid.append(entry)

        # Deduplicate and resolve nesting
        spans = remove_exact_duplicates(spans)
        spans = resolve_nested_spans(spans)

        validated_records.append({
            "id":                  rid,
            "source":             "viocd",
            "task":               "complaint_span_extraction",
            "split":             rec["split"],
            "text":              rec["text"],
            "review_level_label": 1,
            "spans":             spans,
        })

    # ── 4. Save validated records ────────────────────────
    print("\n" + "-" * 70)
    print("4. SAVING VALIDATED RECORDS")
    print("-" * 70)
    valid_path = OUTPUT_DIR / "pilot_manual_validated.jsonl"
    with open(valid_path, "w", encoding="utf-8") as f:
        for rec in validated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {valid_path.name}: {len(validated_records)} records")

    # ── 5. Save invalid entries ──────────────────────────
    print("\n" + "-" * 70)
    print("5. SAVING INVALID ENTRIES")
    print("-" * 70)
    if all_invalid:
        invalid_path = OUTPUT_DIR / "pilot_manual_invalid.csv"
        fieldnames = ["record_id", "split", "reason", "span_index", "detail", "text"]
        with open(invalid_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_invalid)
        print(f"  Saved {invalid_path.name}: {len(all_invalid)} rows")
    else:
        print("  No invalid entries — skipping CSV.")

    # ── 6. Save manifest ─────────────────────────────────
    print("\n" + "-" * 70)
    print("6. SAVING MANIFEST")
    print("-" * 70)
    total_spans = sum(len(r["spans"]) for r in validated_records)
    records_with_spans = sum(1 for r in validated_records if r["spans"])
    records_without_spans = sum(1 for r in validated_records if not r["spans"])
    avg_spans = total_spans / len(validated_records) if validated_records else 0
    split_dist = Counter(r["split"] for r in validated_records)

    manifest = {
        "pilot_records":            len(validated_records),
        "records_with_spans":      records_with_spans,
        "records_without_spans":   records_without_spans,
        "total_spans":            total_spans,
        "average_spans_per_record": round(avg_spans, 3),
        "invalid_span_text_count": error_counts["invalid_span_text"],
        "json_parse_error_count": error_counts["json_parse_error"],
        "text_not_in_review_count": error_counts["text_not_in_review"],
        "missing_output_records":  error_counts["missing_output"],
        "split_distribution":     dict(split_dist),
    }

    manifest_path = OUTPUT_DIR / "pilot_manual_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  Saved {manifest_path.name}")

    # ── 7. Summary ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  1. Pilot records:                   {manifest['pilot_records']}")
    print(f"  2. Records with at least one span: {manifest['records_with_spans']}")
    print(f"  3. Total extracted spans:           {manifest['total_spans']}")
    print(f"  4. Average spans per record:        {manifest['average_spans_per_record']}")
    print(f"  5. Invalid span texts:              {manifest['invalid_span_text_count']}")
    print(f"  6. Text-not-found errors:           {manifest['text_not_in_review_count']}")
    print(f"  7. JSON parse errors:              {manifest['json_parse_error_count']}")
    print(f"  8. Missing output records:          {manifest['missing_output_records']}")

    print(f"\n  [Saved Output Paths]")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".csv", ".json"):
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  ✅ Validation complete.")
    print(f"\n  Next step: Run Step 3B to process validated spans into BIO format.")


if __name__ == "__main__":
    main()
