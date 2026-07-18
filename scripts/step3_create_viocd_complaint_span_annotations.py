#!/usr/bin/env python3
"""
Step 3: Create Complaint Span Annotations for ViOCD
===================================================
Extends ViOCD from review-level complaint detection to complaint span extraction.

Pipeline:
  1. Load ViOCD complaint detection files (Step 2 output)
  2. Select positive reviews (label == 1)
  3. Generate annotation prompts
  4. Run LLM annotation (if API key available)
     - Auto-detects: OpenAI / Anthropic / Google / Azure / Together
     - Falls back to generating prompts for external annotation
  5. Validate character offsets
  6. Repair offset mismatches via exact string matching
  7. Remove invalid spans
  8. Deduplicate and resolve overlaps
  9. Combine with negative all-O records
  10. Save final annotated dataset + manifest

Important:
  - Aspect spans from UIT-ViSD4SA / CausaSent are NOT used as complaint spans.
  - Complaint spans are extracted ONLY from ViOCD positive reviews.
"""

import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_IN  = REPO_ROOT / "data" / "bio_splits" / "complaint_detection"
DATA_OUT = REPO_ROOT / "data" / "complaint_span_annotations"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# Annotation settings
ANNOTATION_BATCH_SIZE = 50       # records per API call
ANNOTATION_DELAY = 1.0            # seconds between API calls (rate limit safety)
ANNOTATE_WITH_LLM = True          # set False to only generate prompts

# ─────────────────────────────────────────────────────────
# Annotation prompt template
# ─────────────────────────────────────────────────────────
ANNOTATION_SYSTEM_PROMPT = (
    "You are annotating Vietnamese customer reviews for complaint span extraction.\n\n"
    "Task:\n"
    "Extract all complaint spans from the given review.\n\n"
    "Definition:\n"
    "A complaint span is a contiguous segment of text that directly expresses "
    "dissatisfaction, service/product failure, inconvenience, unmet expectation, "
    "or a negative customer experience.\n\n"
    "Rules:\n"
    "1. Extract only exact substrings from the original review.\n"
    "2. Do not paraphrase.\n"
    "3. Do not normalize spelling.\n"
    "4. Do not infer implicit complaints.\n"
    "5. Do not extract aspect terms alone unless they include a complaint expression.\n"
    "6. Prefer minimal but complete spans.\n"
    "7. Multiple spans are allowed.\n"
    "8. Return an empty list if no explicit complaint span exists."
)

ANNOTATION_USER_TEMPLATE = (
    "Review ID: {id}\n\n"
    "Review text:\n{text}\n\n"
    "Return only valid JSON:\n"
    '{{\n'
    '  "id": "{id}",\n'
    '  "complaint_spans": [\n'
    '    {{\n'
    '      "text": "...",\n'
    '      "start": 0,\n'
    '      "end": 10\n'
    '    }}\n'
    '  ]\n'
    '}}'
)


# ─────────────────────────────────────────────────────────
# LLM Clients
# ─────────────────────────────────────────────────────────

def try_import_openai() -> Optional[Any]:
    """Try to import and return OpenAI client."""
    try:
        import openai
        return openai
    except ImportError:
        return None


def try_import_anthropic() -> Optional[Any]:
    """Try to import and return Anthropic client."""
    try:
        import anthropic
        return anthropic
    except ImportError:
        return None


def try_import_together() -> Optional[Any]:
    """Try to import and return Together AI client."""
    try:
        import together
        return together
    except ImportError:
        return None


def detect_llm_client():
    """Auto-detect available LLM API and return (client, provider_name, model)."""
    api_key = os.environ.get

    # Check OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        client = try_import_openai()
        if client:
            return client, "openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Check Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        client = try_import_anthropic()
        if client:
            return client, "anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Check Together AI
    if os.environ.get("TOGETHER_API_KEY"):
        client = try_import_together()
        if client:
            return client, "together", os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

    return None, None, None


def build_annotation_prompt(record: Dict) -> Dict:
    """Build a prompt dict for a single record."""
    user_text = ANNOTATION_USER_TEMPLATE.format(id=record["id"], text=record["text"])
    return {
        "id":       record["id"],
        "system":   ANNOTATION_SYSTEM_PROMPT,
        "user":     user_text,
    }


# ─────────────────────────────────────────────────────────
# LLM annotation functions
# ─────────────────────────────────────────────────────────

def annotate_openai(client, model: str, prompts: List[Dict]) -> List[Dict]:
    """Annotate records using OpenAI API."""
    messages = [
        [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]
        for p in prompts
    ]
    # Batch all in one call if possible, otherwise loop
    results = []
    for msg in messages:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=msg,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            results.append(json.loads(content))
        except Exception as e:
            print(f"    [WARN] OpenAI API error: {e}")
            results.append({"id": msg[1]["content"].split('\n')[0].split(': ')[1], "complaint_spans": []})
        time.sleep(ANNOTATION_DELAY)
    return results


def annotate_anthropic(client, model: str, prompts: List[Dict]) -> List[Dict]:
    """Annotate records using Anthropic API."""
    results = []
    for p in prompts:
        try:
            response = client.messages.create(
                model=model,
                system=p["system"],
                messages=[{"role": "user", "content": p["user"]}],
                temperature=0.1,
                max_tokens=2048,
            )
            content = response.content[0].text
            results.append(json.loads(content))
        except Exception as e:
            print(f"    [WARN] Anthropic API error: {e}")
            results.append({"id": p["id"], "complaint_spans": []})
        time.sleep(ANNOTATION_DELAY)
    return results


def annotate_together(client, model: str, prompts: List[Dict]) -> List[Dict]:
    """Annotate records using Together AI API."""
    results = []
    for p in prompts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": p["system"]},
                    {"role": "user", "content": p["user"]},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            results.append(json.loads(content))
        except Exception as e:
            print(f"    [WARN] Together API error: {e}")
            results.append({"id": p["id"], "complaint_spans": []})
        time.sleep(ANNOTATION_DELAY)
    return results


def run_llm_annotation(prompts: List[Dict], batch_size: int = ANNOTATION_BATCH_SIZE) -> List[Dict]:
    """
    Run LLM annotation across all prompts.
    Returns list of {id, complaint_spans} dicts.
    """
    client, provider, model = detect_llm_client()
    if client is None:
        print("  [INFO] No LLM API key detected. Skipping annotation.")
        return []

    print(f"  Using {provider} model: {model}")
    all_results = []

    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    total = len(batches)
    print(f"  Running {total} annotation batches...")

    for batch_idx, batch in enumerate(batches):
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1}/{total}")

        if provider == "openai":
            results = annotate_openai(client, model, batch)
        elif provider == "anthropic":
            results = annotate_anthropic(client, model, batch)
        elif provider == "together":
            results = annotate_together(client, model, batch)
        else:
            results = []

        all_results.extend(results)

    return all_results


# ─────────────────────────────────────────────────────────
# Span validation and repair
# ─────────────────────────────────────────────────────────

def validate_span(span: Dict, text: str) -> Tuple[bool, str]:
    """Validate a single span. Returns (is_valid, reason)."""
    try:
        start = int(span.get("start", -1))
        end   = int(span.get("end",   -1))
    except (ValueError, TypeError):
        return False, f"non-integer offsets: start={span.get('start')}, end={span.get('end')}"

    if start < 0:
        return False, f"start={start} < 0"
    if end > len(text):
        return False, f"end={end} > len(text)={len(text)}"
    if start >= end:
        return False, f"start={start} >= end={end}"
    if start >= len(text):
        return False, f"start={start} >= len(text)={len(text)}"

    extracted = span.get("text", "")
    actual = text[start:end]
    if actual != extracted:
        return False, f"text mismatch: expected {repr(actual)}, got {repr(extracted)}"

    return True, ""


def find_span_in_text(text: str, predicted_text: str) -> Optional[Tuple[int, int]]:
    """
    Find the position of predicted_text inside text using exact string matching.
    Returns (start, end) if found, None otherwise.
    """
    if not predicted_text:
        return None

    idx = text.find(predicted_text)
    if idx >= 0:
        return (idx, idx + len(predicted_text))

    # Try case-insensitive search
    text_lower = text.lower()
    pred_lower = predicted_text.lower()
    idx = text_lower.find(pred_lower)
    if idx >= 0:
        return (idx, idx + len(predicted_text))

    return None


def repair_span(span: Dict, text: str, predicted_start: int) -> Optional[Dict]:
    """
    Attempt to repair an offset-mismatched span via exact string matching.
    Returns repaired span dict or None if repair fails.
    """
    span_text = span.get("text", "")
    if not span_text:
        return None

    # Find all occurrences of span_text in text
    occurrences = []
    search_start = 0
    while True:
        idx = text.find(span_text, search_start)
        if idx < 0:
            break
        occurrences.append((idx, idx + len(span_text)))
        search_start = idx + 1

    if not occurrences:
        return None

    # Choose the occurrence closest to the predicted start
    closest = min(occurrences, key=lambda o: abs(o[0] - predicted_start))
    return {
        "start": closest[0],
        "end":   closest[1],
        "text":  span_text,
    }


def remove_exact_duplicates(spans: List[Dict]) -> List[Dict]:
    """Remove spans that are exact duplicates (same start, end, text)."""
    seen = set()
    unique = []
    for s in spans:
        key = (s["start"], s["end"], s["text"])
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def resolve_overlaps(spans: List[Dict], text: str) -> List[Dict]:
    """
    Resolve overlapping spans.
    Strategy:
      - If one span is fully contained within another: keep the longer one.
      - If spans partially overlap and one is much longer (>=2x): keep the longer one.
      - Otherwise keep both (different complaint expressions).
    """
    if len(spans) <= 1:
        return spans

    # Sort by start, then by length (shorter first)
    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"] - s["start"]))
    resolved = []
    resolved_ranges = []

    for span in sorted_spans:
        s_start, s_end = span["start"], span["end"]
        is_contained = False
        replace_idx = -1

        for i, (r_start, r_end) in enumerate(resolved_ranges):
            # Check containment
            if s_start >= r_start and s_end <= r_end:
                # current span is fully inside resolved span
                is_contained = True
                break
            if r_start >= s_start and r_end <= s_end:
                # resolved span is fully inside current span
                replace_idx = i
                break
            # Check significant overlap
            overlap_len = min(s_end, r_end) - max(s_start, r_start)
            if overlap_len > 0:
                s_len = s_end - s_start
                r_len = r_end - r_start
                # If one is at least 2x the other and contains the other's meaningful content
                if s_len >= 2 * r_len or r_len >= 2 * s_len:
                    # Keep the shorter if it adds meaningful content not in the longer
                    # For now, prefer the shorter span for precision
                    if s_len < r_len:
                        # Current is shorter - check if resolved is much longer
                        if r_len >= 2 * s_len:
                            # Replace the longer with the shorter (more precise)
                            replace_idx = i
                        else:
                            is_contained = True
                    break

        if is_contained:
            continue  # skip this span
        if replace_idx >= 0:
            resolved[replace_idx] = span
            resolved_ranges[replace_idx] = (s_start, s_end)
        else:
            resolved.append(span)
            resolved_ranges.append((s_start, s_end))

    return resolved


# ─────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────

def load_viocd_detection_data() -> Tuple[List[Dict], List[Dict]]:
    """Load ViOCD detection data. Returns (all_records, positive_only)."""
    all_records = []
    for split in ["train", "valid", "test"]:
        path = DATA_IN / f"{split}.jsonl"
        if not path.exists():
            print(f"  [WARN] {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["_split"] = split
                all_records.append(rec)
    positive = [r for r in all_records if r["label"] == 1]
    return all_records, positive


def save_candidates(positive_records: List[Dict]):
    """Save positive complaint candidates by split."""
    by_split = defaultdict(list)
    for rec in positive_records:
        by_split[rec["_split"]].append({
            "id":    rec["id"],
            "split": rec["_split"],
            "text":  rec["text"],
            "label": rec["label"],
        })

    for split in ["train", "valid", "test"]:
        path = DATA_OUT / f"viocd_complaint_candidates_{split}.jsonl"
        recs = by_split.get(split, [])
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if recs:
            print(f"  Saved {path.name}: {len(recs)} candidates")


def save_annotation_prompts(positive_records: List[Dict]):
    """Save annotation prompts for all positive records."""
    path = DATA_OUT / "annotation_prompts.jsonl"
    prompts = [build_annotation_prompt(rec) for rec in positive_records]
    with open(path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved {path.name}: {len(prompts)} prompts")
    return prompts


def run_annotation(prompts: List[Dict]) -> List[Dict]:
    """Run LLM annotation. Returns raw annotation results."""
    path = DATA_OUT / "annotation_results_raw.jsonl"
    results = run_llm_annotation(prompts)
    if results:
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {path.name}: {len(results)} results")
    return results


def validate_and_repair_spans(
    positive_records: List[Dict],
    raw_results: List[Dict],
) -> Tuple[List[Dict], List[Dict], int, int, int]:
    """
    Validate, repair, and clean spans from LLM output.
    Returns (validated_records, invalid_log, repaired_count, invalid_count, valid_count).
    """
    # Build lookup: id → record
    rec_by_id = {rec["id"]: rec for rec in positive_records}
    # Build lookup: id → raw annotation
    ann_by_id = {r.get("id"): r for r in raw_results}

    validated_records = []
    invalid_log = []
    repaired_count = 0
    invalid_count = 0
    valid_count = 0

    for rec in positive_records:
        rid = rec["id"]
        text = rec["text"]

        raw_ann = ann_by_id.get(rid, {})
        raw_spans = raw_ann.get("complaint_spans", [])
        if not isinstance(raw_spans, list):
            raw_spans = []

        clean_spans = []
        for span in raw_spans:
            if not isinstance(span, dict):
                continue

            is_valid, reason = validate_span(span, text)
            if is_valid:
                clean_spans.append({
                    "start":            int(span["start"]),
                    "end":              int(span["end"]),
                    "text":             span["text"],
                    "span_type":        "complaint",
                    "offset_repaired":  False,
                })
                valid_count += 1
                continue

            # Try to repair offset
            try:
                pred_start = int(span.get("start", -1))
            except (ValueError, TypeError):
                pred_start = -1

            repaired = repair_span(span, text, pred_start)
            if repaired is not None:
                # Verify repaired span
                is_valid_after, reason_after = validate_span(repaired, text)
                if is_valid_after:
                    clean_spans.append({
                        "start":            repaired["start"],
                        "end":              repaired["end"],
                        "text":             repaired["text"],
                        "span_type":        "complaint",
                        "offset_repaired":  True,
                    })
                    valid_count += 1
                    repaired_count += 1
                    continue

            # Cannot repair — log and skip
            invalid_count += 1
            invalid_log.append({
                "record_id":    rid,
                "start":        span.get("start", ""),
                "end":          span.get("end", ""),
                "text":         span.get("text", ""),
                "error":        reason,
            })

        # Remove duplicates
        clean_spans = remove_exact_duplicates(clean_spans)

        # Resolve overlaps
        clean_spans = resolve_overlaps(clean_spans, text)

        validated_records.append({
            "id":                  rid,
            "source":              "viocd",
            "task":                "complaint_span_extraction",
            "split":              rec["_split"],
            "text":               text,
            "review_level_label": 1,
            "spans":              clean_spans,
        })

    return validated_records, invalid_log, repaired_count, invalid_count, valid_count


def build_all_records(
    validated_positive: List[Dict],
    all_negative: List[Dict],
) -> Tuple[List[Dict], Dict]:
    """
    Combine validated positive records and negative all-O records.
    Returns (all_records, stats).
    """
    # Negative records → all-O format
    negative_records = []
    for rec in all_negative:
        negative_records.append({
            "id":                  rec["id"],
            "source":              "viocd",
            "task":                "complaint_span_extraction",
            "split":              rec["_split"],
            "text":               rec["text"],
            "review_level_label":  rec["label"],
            "spans":              [],
        })

    all_records = validated_positive + negative_records

    # Stats
    stats = {
        "total_records":              len(all_records),
        "positive_review_records":   len(validated_positive),
        "negative_all_o_records":    len(negative_records),
        "total_complaint_spans":     sum(len(r["spans"]) for r in validated_positive),
        "records_with_spans":        sum(1 for r in validated_positive if r["spans"]),
        "records_without_spans":     sum(1 for r in validated_positive if not r["spans"]),
    }

    return all_records, stats


def save_splits(all_records: List[Dict]):
    """Save train/valid/test JSONL files."""
    by_split = defaultdict(list)
    for rec in all_records:
        by_split[rec["split"]].append(rec)

    for split in ["train", "valid", "test"]:
        path = DATA_OUT / f"{split}.jsonl"
        recs = by_split.get(split, [])
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Saved {path.name}: {len(recs)} records")


def save_manifest(
    stats: Dict,
    validated_positive: List[Dict],
    repaired_count: int,
    invalid_count: int,
):
    """Save complaint_span_manifest.json."""
    # Per-split breakdown
    split_stats = defaultdict(lambda: {
        "records": 0, "positive": 0, "negative": 0,
        "spans": 0, "records_with_spans": 0, "records_without_spans": 0,
    })
    for rec in validated_positive:
        s = rec["split"]
        split_stats[s]["records"] += 1
        split_stats[s]["positive"] += 1
        split_stats[s]["spans"] += len(rec["spans"])
        if rec["spans"]:
            split_stats[s]["records_with_spans"] += 1
        else:
            split_stats[s]["records_without_spans"] += 1

    manifest = {
        "total_records":            stats["total_records"],
        "positive_review_records":  stats["positive_review_records"],
        "negative_all_o_records":   stats["negative_all_o_records"],
        "total_complaint_spans":    stats["total_complaint_spans"],
        "records_with_spans":       stats["records_with_spans"],
        "records_without_spans":    stats["records_without_spans"],
        "offset_repaired_spans":    repaired_count,
        "invalid_removed_spans":    invalid_count,
        "splits": {
            split: dict(split_stats[split])
            for split in ["train", "valid", "test"]
            if split in split_stats
        },
    }

    path = DATA_OUT / "complaint_span_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  Saved {path.name}")

    return manifest


def save_invalid_spans(invalid_log: List[Dict]):
    """Save invalid spans to CSV."""
    if not invalid_log:
        return
    path = DATA_OUT / "invalid_complaint_spans.csv"
    fieldnames = ["record_id", "start", "end", "text", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(invalid_log)
    print(f"  Saved {path.name}: {len(invalid_log)} rows")


def simulate_annotation(positive_records: List[Dict]) -> List[Dict]:
    """
    Fallback: simulate annotation for testing.
    Returns empty complaint_spans for all records (no LLM needed).
    This allows the pipeline to proceed through validation without real annotations.
    """
    return [
        {"id": rec["id"], "complaint_spans": []}
        for rec in positive_records
    ]


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Step 3: Create Complaint Span Annotations for ViOCD")
    print("=" * 70)
    print(f"\n  Input:  {DATA_IN}")
    print(f"  Output: {DATA_OUT}\n")

    # ── Step 1: Load ViOCD data ───────────────────────────
    print("-" * 70)
    print("1. LOADING VIOCD COMPLAINT DETECTION DATA")
    print("-" * 70)
    all_records, positive_records = load_viocd_detection_data()
    negative_records = [r for r in all_records if r["label"] == 0]
    print(f"  Total records: {len(all_records)}")
    print(f"  Positive (label=1): {len(positive_records)}")
    print(f"  Negative (label=0): {len(negative_records)}")

    by_split_pos = Counter(r["_split"] for r in positive_records)
    by_split_neg = Counter(r["_split"] for r in negative_records)
    print(f"\n  By split:")
    for split in ["train", "valid", "test"]:
        print(f"    {split}: {by_split_pos.get(split, 0)} positive, {by_split_neg.get(split, 0)} negative")

    # ── Step 2: Save positive candidates ─────────────────
    print("\n" + "-" * 70)
    print("2. SAVING POSITIVE COMPLAINT CANDIDATES")
    print("-" * 70)
    save_candidates(positive_records)

    # ── Step 3: Build annotation prompts ─────────────────
    print("\n" + "-" * 70)
    print("3. BUILDING ANNOTATION PROMPTS")
    print("-" * 70)
    prompts = save_annotation_prompts(positive_records)

    # ── Step 4: Run LLM annotation ───────────────────────
    print("\n" + "-" * 70)
    print("4. RUNNING LLM ANNOTATION")
    print("-" * 70)

    client, provider, model = detect_llm_client()
    if client is not None:
        print(f"  Detected: {provider} / {model}")
        raw_results = run_annotation(prompts)
    else:
        print("  [INFO] No LLM API key detected.")
        print("  Generating annotation_prompts.jsonl for external annotation.")
        print("  To annotate with an LLM, set one of:")
        print("    OPENAI_API_KEY      → GPT-4o / GPT-4o-mini")
        print("    ANTHROPIC_API_KEY   → Claude Sonnet / Haiku")
        print("    TOGETHER_API_KEY    → Llama-3.3-70B")
        print()
        print("  Falling back to simulation (empty spans) to demonstrate validation pipeline.")
        raw_results = simulate_annotation(positive_records)

    # ── Step 5: Validate and repair spans ─────────────────
    print("\n" + "-" * 70)
    print("5. VALIDATING AND REPAIRING SPANS")
    print("-" * 70)

    validated_positive, invalid_log, repaired_count, invalid_count, valid_count = (
        validate_and_repair_spans(positive_records, raw_results)
    )

    print(f"  Total complaint spans generated:  {len(raw_results)} records processed")
    print(f"  Valid spans:                    {valid_count}")
    print(f"  Offset repaired spans:          {repaired_count}")
    print(f"  Invalid spans removed:           {invalid_count}")

    if invalid_log:
        print(f"\n  Examples of invalid spans:")
        for item in invalid_log[:5]:
            print(f"    {item['record_id']}: [{item['start']},{item['end']}] "
                  f"text={repr(item['text'][:30])} → {item['error']}")

    # ── Step 6: Save invalid spans ───────────────────────
    print("\n" + "-" * 70)
    print("6. SAVING INVALID SPANS")
    print("-" * 70)
    save_invalid_spans(invalid_log)

    # ── Step 7: Build all records ────────────────────────
    print("\n" + "-" * 70)
    print("7. BUILDING COMBINED DATASET")
    print("-" * 70)
    all_records_combined, stats = build_all_records(validated_positive, negative_records)
    print(f"  Positive validated records:  {stats['positive_review_records']}")
    print(f"  Negative all-O records:     {stats['negative_all_o_records']}")
    print(f"  Total combined records:     {stats['total_records']}")
    print(f"  Total complaint spans:      {stats['total_complaint_spans']}")

    # ── Step 8: Save final splits ───────────────────────
    print("\n" + "-" * 70)
    print("8. SAVING FINAL SPLITS")
    print("-" * 70)
    save_splits(all_records_combined)

    # ── Step 9: Save manifest ───────────────────────────
    print("\n" + "-" * 70)
    print("9. SAVING MANIFEST")
    print("-" * 70)
    manifest = save_manifest(stats, validated_positive, repaired_count, invalid_count)

    # ── Final summary ───────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Positive ViOCD candidates:")
    for split in ["train", "valid", "test"]:
        cnt = by_split_pos.get(split, 0)
        print(f"    {split}: {cnt}")
    print(f"\n  Negative all-O records:")
    for split in ["train", "valid", "test"]:
        cnt = by_split_neg.get(split, 0)
        print(f"    {split}: {cnt}")
    print(f"\n  Total complaint spans:        {stats['total_complaint_spans']}")
    print(f"  Valid spans:                  {valid_count}")
    print(f"  Repaired spans:               {repaired_count}")
    print(f"  Invalid spans removed:        {invalid_count}")
    print(f"  Records with spans:           {stats['records_with_spans']}")
    print(f"  Records without spans:        {stats['records_without_spans']}")

    print(f"\n  [Saved Output Paths]")
    for f in sorted(DATA_OUT.iterdir()):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    if client is None:
        print(f"\n  ⚠️  No LLM API key set.")
        print(f"  ⚠️  annotation_prompts.jsonl saved. Run annotation externally, then re-run this script.")
        print(f"  ⚠️  Using simulated empty spans for demonstration.")
        sys.exit(0)
    else:
        print(f"\n  ✅ Step 3 complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
