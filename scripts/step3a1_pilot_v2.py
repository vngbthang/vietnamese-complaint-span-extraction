#!/usr/bin/env python3
"""
Step 3A.1: Improved Pilot LLM Annotation for ViOCD Complaint Spans
=================================================================
- Improved prompt with exact-copy rules and XML delimiters
- Fixed validation (empty [] = valid)
- Safe post-processing (trim wrapping chars, trim terminal punctuation)
- No fuzzy matching, no edit distance

Output: data/complaint_span_annotations/pilot_v2/
"""

import csv
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_IN   = REPO_ROOT / "data" / "complaint_span_annotations"
PILOT_IN  = DATA_IN  / "pilot"
DATA_OUT  = DATA_IN  / "pilot_v2"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# API config
# ─────────────────────────────────────────────────────────
NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "mistralai/mistral-small-4-119b-2603"
API_DELAY = 0.3  # seconds

# ─────────────────────────────────────────────────────────
# Improved annotation prompt
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are annotating Vietnamese customer reviews for complaint span extraction.

A complaint span is an exact contiguous substring from the review that directly expresses dissatisfaction, product/service failure, inconvenience, unmet expectation, or negative customer experience.

Critical copying rule:
Every returned span text must be copied exactly from the review. Do not add, remove, rewrite, normalize, correct spelling, change punctuation, or change spacing. The output span must satisfy: span_text in review_text.

Annotation rules:
1. Extract exact substrings only.
2. Do not paraphrase.
3. Do not normalize spelling.
4. Do not correct typos.
5. Do not add punctuation.
6. Do not remove punctuation if it is part of the copied span.
7. Do not extract aspect terms alone.
8. Prefer minimal but complete complaint expressions.
9. Multiple spans are allowed.
10. Return [] if there is no explicit complaint expression or if you are unsure.
11. Before final output, verify each span appears exactly in the review.

Return JSON only."""

USER_PROMPT_TEMPLATE = """Review ID:
{id}

Review text between <review> and </review>:
<review>
{text}
</review>

Return exactly:
{{
  "id": "{id}",
  "complaint_spans": [
    {{"text": "exact substring copied from the review"}}
  ]
}}"""


# ─────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────

def call_api(record: Dict) -> Tuple[str, Optional[str]]:
    """
    Call NVIDIA API for one record.
    Returns (raw_response_text, error_message).
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(id=record["id"], text=record["text"])

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(
            NVIDIA_BASE_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=60,
        )

        if resp.status_code != 200:
            err = resp.json().get("error", {}) or {}
            msg = err.get("message", f"HTTP {resp.status_code}")
            return "", msg

        content = resp.json()["choices"][0]["message"]["content"]
        return content, None

    except requests.exceptions.Timeout:
        return "", "timeout"
    except Exception as e:
        return "", str(e)


def extract_json(content: str) -> Optional[dict]:
    """
    Extract and parse JSON from LLM response.
    Handles markdown code fences and partial responses.
    """
    content = content.strip()

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    code_lines = []
    in_code = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)

    if code_lines:
        try:
            return json.loads("\n".join(code_lines).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object or array by bracket matching
    for start_char, start_idx in [("[", content.find("[")), ("{", content.find("{"))]:
        if start_idx == -1:
            continue
        bracket = start_char
        close = "]" if bracket == "[" else "}"
        depth = 0
        end_idx = start_idx
        in_string = False
        for i, c in enumerate(content[start_idx:], start_idx):
            if not in_string:
                if c == '"':
                    in_string = True
                elif c == bracket:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            else:
                if c == '"' and (i == 0 or content[i - 1] != "\\"):
                    in_string = False
        try:
            return json.loads(content[start_idx:end_idx])
        except json.JSONDecodeError:
            continue

    return None


# ─────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────

def trim_wrapping_chars(text: str) -> str:
    """Trim leading/trailing whitespace and common wrapping characters."""
    return text.strip(" \t\n\r\x0b\x0c\u00a0\"'`")


def trim_terminal_punctuation(text: str) -> str:
    """Remove trailing common terminal punctuation."""
    return text.rstrip(".,;:!?…\u2026")


def safe_repair(span_text: str, review_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try safe repairs on span_text to find exact match in review_text.
    Returns (repaired_text, repair_type) or (None, None).
    Repair types:
      - "trim_wrapping_chars"
      - "trim_extra_terminal_punctuation"
    """
    # A. Exact match — no repair needed
    if span_text in review_text:
        return span_text, None

    # B. Trim wrapping characters
    trimmed = trim_wrapping_chars(span_text)
    if trimmed in review_text:
        return trimmed, "trim_wrapping_chars"

    # C. Trim extra terminal punctuation
    for _ in range(3):  # up to 3 trailing punctuation chars
        trimmed2 = trim_terminal_punctuation(trimmed)
        if trimmed2 in review_text:
            return trimmed2, "trim_extra_terminal_punctuation"
        if trimmed2 == trimmed:
            break
        trimmed = trimmed2

    return None, None


# ─────────────────────────────────────────────────────────
# Validation and post-processing
# ─────────────────────────────────────────────────────────

def validate_record(record: Dict, raw_response: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Validate and post-process LLM output for one record.
    Returns (accepted_spans, invalid_entries, stats).
    """
    rid = record["id"]
    text = record["text"]
    accepted = []
    invalid = []
    stats = {
        "empty_span": 0,
        "span_not_dict": 0,
        "trim_wrapping_repairs": 0,
        "terminal_punct_repairs": 0,
        "text_not_found": 0,
        "json_invalid": 0,
    }

    # Parse JSON
    parsed = extract_json(raw_response)

    if parsed is None:
        stats["json_invalid"] += 1
        invalid.append({
            "id": rid,
            "review_text": text,
            "predicted_span_text": "",
            "reason": "json_invalid",
        })
        return [], invalid, stats

    # Extract complaint_spans
    if isinstance(parsed, dict):
        complaint_spans = parsed.get("complaint_spans", [])
    elif isinstance(parsed, list) and len(parsed) > 0:
        # Wrap single object
        complaint_spans = parsed[0].get("complaint_spans", []) if isinstance(parsed[0], dict) else []
    else:
        complaint_spans = []

    # Handle empty
    if not isinstance(complaint_spans, list):
        stats["span_not_dict"] += 1
        invalid.append({
            "id": rid,
            "review_text": text,
            "predicted_span_text": str(complaint_spans)[:100],
            "reason": "complaint_spans_not_list",
        })
        return [], invalid, stats

    if len(complaint_spans) == 0:
        stats["empty_span"] += 1
        # Empty is valid — no span added, no invalid entry
        return [], invalid, stats

    # Process each span
    for idx, span in enumerate(complaint_spans):
        if not isinstance(span, dict):
            stats["span_not_dict"] += 1
            invalid.append({
                "id": rid,
                "review_text": text,
                "predicted_span_text": str(span)[:100],
                "reason": "span_not_dict",
            })
            continue

        span_text = span.get("text", "")

        if not isinstance(span_text, str) or not span_text.strip():
            stats["empty_span"] += 1
            invalid.append({
                "id": rid,
                "review_text": text,
                "predicted_span_text": repr(span_text)[:50],
                "reason": "empty_span",
            })
            continue

        # Try repairs
        repaired_text, repair_type = safe_repair(span_text, text)

        if repaired_text is None:
            stats["text_not_found"] += 1
            invalid.append({
                "id": rid,
                "review_text": text,
                "predicted_span_text": span_text[:200],
                "reason": "text_not_in_review",
            })
            continue

        if repair_type == "trim_wrapping_chars":
            stats["trim_wrapping_repairs"] += 1
        elif repair_type == "trim_extra_terminal_punctuation":
            stats["terminal_punct_repairs"] += 1

        # Compute offset
        start = text.find(repaired_text)
        accepted.append({
            "text":         repaired_text,
            "start":        start,
            "end":          start + len(repaired_text),
            "repair_type":  repair_type,
        })

    return accepted, invalid, stats


def deduplicate_spans(spans: List[Dict]) -> List[Dict]:
    """Remove exact-duplicate spans."""
    seen, unique = set(), []
    for s in spans:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    return unique


def resolve_nested_spans(spans: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove spans fully contained within other spans.
    Returns (kept_spans, removed_spans).
    """
    if len(spans) <= 1:
        return spans, []

    # Sort by length ascending (prefer keeping shorter when conflicting)
    sorted_spans = sorted(spans, key=lambda s: (s["end"] - s["start"], s["start"]))
    kept = []
    removed = []

    for span in sorted_spans:
        s_start, s_end = span["start"], span["end"]

        # Is this span dominated by an existing kept span?
        dominated = any(
            e["start"] <= s_start and s_end <= e["end"]
            for e in kept
        )
        if dominated:
            removed.append({**span, "reason": "nested_span_removed"})
            continue

        # Remove any kept spans that are fully contained in this span
        kept = [
            e for e in kept
            if not (s_start <= e["start"] and e["end"] <= s_end)
        ]
        kept.append(span)

    return kept, removed


# ─────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────

def load_pilot_candidates() -> List[Dict]:
    """Load the same 100 pilot candidates from Step 3A."""
    path = PILOT_IN / "pilot_candidates.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} pilot candidates from {path.name}")
    return records


def run_annotation(records: List[Dict]) -> Dict[str, Dict]:
    """Run annotation for all records. Returns dict mapping id -> raw output."""
    print(f"\n  Model: {MODEL}")
    print(f"  Total records: {len(records)}")

    outputs = {}
    for i, rec in enumerate(records):
        if i == 0 or (i + 1) % 10 == 0 or i == len(records) - 1:
            print(f"  Annotating {i + 1}/{len(records)}...")

        content, error = call_api(rec)
        outputs[rec["id"]] = {
            "raw_response": content,
            "error": error,
        }

        # Save interim every 25 records
        if (i + 1) % 25 == 0:
            with open(DATA_OUT / "pilot_v2_raw_outputs.jsonl", "w", encoding="utf-8") as f:
                for rid, out in outputs.items():
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"    [Saved interim: {len(outputs)} results]")

        time.sleep(API_DELAY)

    return outputs


def run_validation(records: List[Dict], outputs: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
    """Validate all records and post-process."""
    validated = []
    all_invalid = []
    aggregate_stats = Counter()

    for rec in records:
        rid = rec["id"]
        output = outputs.get(rid, {})
        raw = output.get("raw_response", "")
        error = output.get("error")

        # Missing output
        if not raw and error:
            aggregate_stats["missing_output"] += 1
            all_invalid.append({
                "id": rid,
                "review_text": rec["text"],
                "predicted_span_text": "",
                "reason": f"missing_output: {error}",
            })
            validated.append({
                "id": rid, "split": rec["split"],
                "text": rec["text"], "spans": [],
            })
            continue

        # Validate
        accepted, invalid, stats = validate_record(rec, raw)
        for k, v in stats.items():
            aggregate_stats[k] += v

        for e in invalid:
            all_invalid.append(e)

        # Deduplicate
        accepted = deduplicate_spans(accepted)

        # Resolve nested
        accepted, removed = resolve_nested_spans(accepted)
        for r in removed:
            all_invalid.append({
                "id": rid,
                "review_text": rec["text"],
                "predicted_span_text": r["text"],
                "reason": r["reason"],
            })

        validated.append({
            "id": rid,
            "split": rec["split"],
            "text": rec["text"],
            "spans": accepted,
        })

    return validated, all_invalid, dict(aggregate_stats)


def save_outputs(validated: List[Dict], invalid: List[Dict], stats: Dict):
    """Save all output files."""

    # Validated
    with open(DATA_OUT / "pilot_v2_validated_outputs.jsonl", "w", encoding="utf-8") as f:
        for rec in validated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Raw
    with open(DATA_OUT / "pilot_v2_raw_outputs.jsonl", "w", encoding="utf-8") as f:
        pass  # already saved incrementally

    # Invalid CSV
    if invalid:
        with open(DATA_OUT / "pilot_v2_invalid_outputs.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "reason", "predicted_span_text", "review_text"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(invalid)

    # Manifest
    total_spans      = sum(len(r["spans"]) for r in validated)
    records_with     = sum(1 for r in validated if r["spans"])
    records_without  = sum(1 for r in validated if not r["spans"])
    avg              = total_spans / len(validated) if validated else 0
    split_dist       = dict(Counter(r["split"] for r in validated))

    manifest = {
        "pilot_records":                       len(validated),
        "records_with_spans":                records_with,
        "records_without_spans":             records_without,
        "total_raw_spans":                  (stats.get("empty_span", 0) + stats.get("trim_wrapping_repairs", 0)
                                           + stats.get("terminal_punct_repairs", 0)
                                           + stats.get("text_not_found", 0)
                                           + stats.get("span_not_dict", 0)
                                           + sum(len(r["spans"]) for r in validated)),
        "total_valid_spans":                total_spans,
        "average_valid_spans_per_record":   round(avg, 3),
        "invalid_json_outputs":              stats.get("json_invalid", 0),
        "span_text_not_found_count":        stats.get("text_not_found", 0),
        "span_not_dict_count":              stats.get("span_not_dict", 0),
        "empty_span_count":                 stats.get("empty_span", 0),
        "trim_wrapping_repairs":            stats.get("trim_wrapping_repairs", 0),
        "terminal_punctuation_repairs":      stats.get("terminal_punct_repairs", 0),
        "nested_span_removed_count":        sum(1 for e in invalid if e.get("reason") == "nested_span_removed"),
        "missing_output_records":           stats.get("missing_output", 0),
        "split_distribution":               split_dist,
        "model":                            MODEL,
    }

    with open(DATA_OUT / "pilot_v2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def print_examples(validated: List[Dict], n: int = 10):
    """Print examples for manual inspection."""
    examples_path = DATA_OUT / "pilot_v2_examples.txt"
    shown = 0

    with open(examples_path, "w", encoding="utf-8") as out:
        out.write("=" * 70 + "\n")
        out.write("Step 3A.1: Pilot Annotation Examples\n")
        out.write("=" * 70 + "\n\n")

        for rec in validated:
            if shown >= n:
                break
            spans = rec["spans"]
            if not spans:
                continue
            shown += 1

            out.write(f"[{shown}] ID: {rec['id']}  ({rec['split']})\n")
            out.write(f"    Review: {repr(rec['text'])}\n")
            out.write(f"    Accepted spans ({len(spans)}):\n")
            for i, span in enumerate(spans[:5]):
                repair_note = f" [repaired: {span.get('repair_type', 'exact')}]" if span.get("repair_type") else ""
                out.write(f"      [{i+1}] [{span['start']},{span['end']}] {repr(span['text'][:60])}{repair_note}\n")
            out.write("\n")

        out.write(f"\nTotal shown: {shown}\n")

    print(f"\n  Saved examples to {examples_path.name}")
    # Also print to stdout
    print(f"\n{'='*70}")
    print(f"10 EXAMPLES FOR MANUAL INSPECTION (saved to pilot_v2_examples.txt)")
    print(f"{'='*70}")
    for rec in validated:
        if shown >= n:
            break
        spans = rec["spans"]
        if not spans:
            continue
        shown += 1
        print(f"\n[{shown}] ID: {rec['id']}  ({rec['split']})")
        print(f"    Review: {repr(rec['text'][:80])}")
        for i, span in enumerate(spans[:5]):
            repair_note = f" [repaired: {span.get('repair_type', 'exact')}]" if span.get("repair_type") else ""
            print(f"    [{i+1}] [{span['start']},{span['end']}] {repr(span['text'][:60])}{repair_note}")


def main():
    print("=" * 70)
    print("Step 3A.1: Improved Pilot Annotation — ViOCD Complaint Spans")
    print("=" * 70)
    print(f"\n  Input candidates: {PILOT_IN / 'pilot_candidates.jsonl'}")
    print(f"  Output folder:    {DATA_OUT}")
    print(f"  Model:           {MODEL}\n")

    # 1. Load same 100 pilot candidates
    print("-" * 70)
    print("1. LOADING PILOT CANDIDATES")
    print("-" * 70)
    records = load_pilot_candidates()

    # 2. Run annotation
    print("\n" + "-" * 70)
    print("2. RUNNING ANNOTATION")
    print("-" * 70)
    outputs = run_annotation(records)

    # Save raw outputs
    with open(DATA_OUT / "pilot_v2_raw_outputs.jsonl", "w", encoding="utf-8") as f:
        for rid, out in outputs.items():
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\n  Saved pilot_v2_raw_outputs.jsonl: {len(outputs)} records")

    # 3. Validate and post-process
    print("\n" + "-" * 70)
    print("3. VALIDATING AND POST-PROCESSING")
    print("-" * 70)
    validated, invalid, stats = run_validation(records, outputs)

    print(f"  Validated records:      {len(validated)}")
    print(f"  Records with spans:    {sum(1 for r in validated if r['spans'])}")
    print(f"  Total valid spans:     {sum(len(r['spans']) for r in validated)}")
    print(f"  Invalid entries:       {len(invalid)}")
    print(f"    json_invalid:        {stats.get('json_invalid', 0)}")
    print(f"    text_not_in_review:  {stats.get('text_not_found', 0)}")
    print(f"    span_not_dict:       {stats.get('span_not_dict', 0)}")
    print(f"    empty_span:          {stats.get('empty_span', 0)}")
    print(f"    trim_wrapping:       {stats.get('trim_wrapping_repairs', 0)}")
    print(f"    trim_terminal_punct: {stats.get('terminal_punct_repairs', 0)}")
    print(f"    missing_output:      {stats.get('missing_output', 0)}")

    # 4. Save outputs
    print("\n" + "-" * 70)
    print("4. SAVING OUTPUTS")
    print("-" * 70)
    manifest = save_outputs(validated, invalid, stats)

    # 5. Examples
    print_examples(validated, n=10)

    # Summary
    print("\n" + "=" * 70)
    print("MANIFEST SUMMARY")
    print("=" * 70)
    print(f"\n  pilot_records:                      {manifest['pilot_records']}")
    print(f"  records_with_spans:               {manifest['records_with_spans']}")
    print(f"  records_without_spans:            {manifest['records_without_spans']}")
    print(f"  total_raw_spans:                 {manifest['total_raw_spans']}")
    print(f"  total_valid_spans:              {manifest['total_valid_spans']}")
    print(f"  average_valid_spans_per_record:  {manifest['average_valid_spans_per_record']}")
    print(f"  invalid_json_outputs:            {manifest['invalid_json_outputs']}")
    print(f"  span_text_not_found_count:       {manifest['span_text_not_found_count']}")
    print(f"  span_not_dict_count:            {manifest['span_not_dict_count']}")
    print(f"  empty_span_count:               {manifest['empty_span_count']}")
    print(f"  trim_wrapping_repairs:          {manifest['trim_wrapping_repairs']}")
    print(f"  terminal_punctuation_repairs:    {manifest['terminal_punctuation_repairs']}")
    print(f"  nested_span_removed_count:      {manifest['nested_span_removed_count']}")
    print(f"  missing_output_records:         {manifest['missing_output_records']}")
    print(f"  split_distribution:            {manifest['split_distribution']}")

    # Success criteria
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA CHECK")
    print("="*70)
    raw = manifest["total_raw_spans"]
    tnf = manifest["span_text_not_found_count"]
    ratio = (tnf / raw * 100) if raw > 0 else 0
    ok_json   = "  OK" if manifest["invalid_json_outputs"] == 0 else "  FAIL"
    ok_miss   = "  OK" if manifest["missing_output_records"] == 0 else "  FAIL"
    ok_notdict= "  OK" if manifest["span_not_dict_count"] == 0 else "  WARN"
    ok_ratio  = f"  OK ({ratio:.1f}%)" if ratio <= 5 else f"  WARN ({ratio:.1f}%)"
    print(f"  invalid_json_outputs = 0:      {ok_json} ({manifest['invalid_json_outputs']})")
    print(f"  missing_output_records = 0:    {ok_miss} ({manifest['missing_output_records']})")
    print(f"  span_not_dict_count = 0:       {ok_notdict} ({manifest['span_not_dict_count']})")
    print(f"  text_not_found / raw <= 5%:   {ok_ratio} ({tnf}/{raw})")

    print(f"\n  [Output files in {DATA_OUT}]")
    for f in sorted(DATA_OUT.iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".csv", ".json", ".txt"):
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  ✅ Step 3A.1 complete.")


if __name__ == "__main__":
    main()
