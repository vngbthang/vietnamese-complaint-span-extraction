#!/usr/bin/env python3
"""
Retry script for Step 3A: Retry failed LLM calls due to rate limits.
Reads pilot_raw_outputs.jsonl, finds records with errors, retries them.
"""

import csv
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

warnings.filterwarnings("ignore")

import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT  = REPO_ROOT / "data" / "complaint_span_annotations" / "pilot"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it before running: export GEMINI_API_KEY='your_key_here'"
    )
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
    f":generateContent?key={GEMINI_API_KEY}"
)
BATCH_SIZE = 5
INITIAL_DELAY = 30  # seconds

SYSTEM_INSTRUCTION = (
    "You are annotating Vietnamese customer reviews for complaint span extraction.\n\n"
    "Task:\n"
    "Extract all complaint spans from the given review.\n\n"
    "Definition:\n"
    "A complaint span is a contiguous text segment that directly expresses "
    "dissatisfaction, product/service failure, inconvenience, unmet expectation, "
    "or negative customer experience.\n\n"
    "Rules:\n"
    "1. Extract only exact substrings from the original review.\n"
    "2. Do not rewrite, normalize, translate, or paraphrase.\n"
    "3. Do not infer implicit complaints.\n"
    "4. Do not extract aspect terms alone unless the phrase itself expresses a complaint.\n"
    "5. Prefer minimal but complete spans.\n"
    "6. Multiple complaint spans are allowed.\n"
    "7. Return an empty list if there is no explicit complaint expression.\n"
    "8. Every returned span text must appear exactly in the original review.\n"
    "9. Return text only — NO character offsets."
)

OUTPUT_FORMAT = (
    "Return ONLY valid JSON array. No explanation, no markdown, no code fences.\n"
    "Format:\n"
    "[\n"
    '  {"id": "record_id", "complaint_spans": [{"text": "..."}]},\n'
    "  ...\n"
    "]"
)


def find_retry_delay(error_msg: str) -> float:
    """Extract retry delay from Gemini error message."""
    match = re.search(r"retry in ([\d.]+)s", error_msg)
    if match:
        return max(float(match.group(1)), 5)
    return INITIAL_DELAY


def call_gemini_batch(records: List[Dict], retry_count: int = 0) -> Tuple[List[Dict], float]:
    """Call Gemini API for a batch. Returns (results, suggested_delay)."""
    user_parts = [OUTPUT_FORMAT, ""]
    for rec in records:
        user_parts.append(f'ID: {rec["id"]}')
        user_parts.append(f'Text: {rec["text"]}')
        user_parts.append("")

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": "\n".join(user_parts)}]
        }],
        "generationConfig": {
            "temperature":  0.1,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
    }

    results = []
    suggested_delay = 5.0

    try:
        resp = requests.post(
            GEMINI_URL, json=payload, timeout=120,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 429:
            err_data = resp.json()
            err_msg  = err_data.get("error", {}).get("message", "")
            delay    = find_retry_delay(err_msg)
            print(f"    [RATE LIMIT] retry in {delay:.0f}s")
            for rec in records:
                results.append({
                    "id": rec["id"],
                    "raw_response": "",
                    "error": f"rate_limit_retry_{retry_count}: {err_msg[:100]}",
                    "retry_after": delay,
                })
            return results, delay

        if resp.status_code != 200:
            err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
            for rec in records:
                results.append({"id": rec["id"], "raw_response": "", "error": err_msg})
            return results, 5.0

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            for rec in records:
                results.append({"id": rec["id"], "raw_response": "", "error": "no_candidates"})
            return results, 5.0

        raw_text = ""
        for part in candidates[0].get("content", {}).get("parts", []):
            if "text" in part:
                raw_text += part["text"]

        parsed = None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = re.sub(r"```json\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = None

        output_list = None
        if isinstance(parsed, list):
            output_list = parsed
        elif isinstance(parsed, dict):
            for key in ["predictions", "results", "annotations", "data"]:
                if key in parsed and isinstance(parsed[key], list):
                    output_list = parsed[key]
                    break
            if output_list is None and "id" in parsed and "complaint_spans" in parsed:
                output_list = [parsed]

        if output_list is None:
            for rec in records:
                results.append({
                    "id": rec["id"],
                    "raw_response": raw_text[:500],
                    "error": "unrecognized_structure",
                })
            return results, 5.0

        result_map = {item.get("id"): item for item in output_list}
        for rec in records:
            rid = rec["id"]
            if rid in result_map:
                results.append({
                    "id": rid,
                    "raw_response": json.dumps(result_map[rid], ensure_ascii=False),
                    "error": None,
                })
            else:
                results.append({
                    "id": rid,
                    "raw_response": "",
                    "error": "record_not_in_response",
                })

    except requests.exceptions.Timeout:
        for rec in records:
            results.append({"id": rec["id"], "raw_response": "", "error": "timeout"})
        suggested_delay = 10.0
    except Exception as e:
        for rec in records:
            results.append({"id": rec["id"], "raw_response": "", "error": str(e)})
        suggested_delay = 10.0

    return results, suggested_delay


def main():
    print("=" * 70)
    print("Step 3A: Retry Failed Records (Rate Limit Recovery)")
    print("=" * 70)

    # Load current results
    raw_path = DATA_OUT / "pilot_raw_outputs.jsonl"
    with open(raw_path, encoding="utf-8") as f:
        existing = {json.loads(line)["id"]: json.loads(line) for line in f}

    # Load candidates for failed IDs
    cand_path = DATA_OUT / "pilot_candidates.jsonl"
    with open(cand_path, encoding="utf-8") as f:
        candidates = {json.loads(line)["id"]: json.loads(line) for line in f}

    # Find failed records
    failed_ids = [
        rid for rid, out in existing.items()
        if out.get("error") and ("quota" in out["error"].lower() or "rate_limit" in out["error"].lower())
    ]
    print(f"  Total results loaded: {len(existing)}")
    print(f"  Rate-limit failed records: {len(failed_ids)}")

    if not failed_ids:
        print("  No rate-limit failures to retry. Done!")
        return

    # Group into batches
    failed_records = [candidates[rid] for rid in failed_ids if rid in candidates]
    print(f"  Records to retry: {len(failed_records)}")

    # Retry with exponential backoff
    MAX_RETRIES = 5
    current_delay = INITIAL_DELAY

    for retry_round in range(1, MAX_RETRIES + 1):
        still_failed = [
            rid for rid, out in existing.items()
            if out.get("error") and ("quota" in str(out.get("error")).lower() or "rate_limit" in str(out.get("error")).lower())
        ]
        if not still_failed:
            print(f"\n  All rate-limit records succeeded on round {retry_round-1}!")
            break

        failed_records = [candidates[rid] for rid in still_failed if rid in candidates]
        print(f"\n  Retry round {retry_round}/{MAX_RETRIES}: {len(failed_records)} records (delay={current_delay:.0f}s)")

        for batch_start in range(0, len(failed_records), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(failed_records))
            batch = failed_records[batch_start:batch_end]

            results, suggested = call_gemini_batch(batch, retry_round - 1)
            current_delay = max(suggested, current_delay)

            for r in results:
                rid = r["id"]
                existing[rid] = r

            # Save incrementally
            with open(raw_path, "w", encoding="utf-8") as f:
                for out in existing.values():
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

            print(f"    Batch {batch_start//BATCH_SIZE + 1}: {len(failed_records)//BATCH_SIZE + 1} records done")
            time.sleep(current_delay)

        current_delay = min(current_delay * 1.5, 120)

    # Final stats
    remaining_errors = sum(
        1 for out in existing.values() if out.get("error")
    )
    rate_limit_errors = sum(
        1 for out in existing.values() if out.get("error") and "rate_limit" in str(out.get("error"))
    )
    print(f"\n  Remaining errors: {remaining_errors}")
    print(f"  Remaining rate-limit: {rate_limit_errors}")
    print(f"  Records updated in: {raw_path}")

    # Re-run validation
    print("\n  Re-running validation...")
    import csv as csv_mod
    from collections import Counter as Counter2

    def find_span_offset(text, span_text):
        if not span_text: return None
        span_text = span_text.strip()
        if not span_text: return None
        idx = text.find(span_text)
        if idx >= 0: return (idx, idx + len(span_text))
        return None

    def is_invalid_span_text(text):
        if not text or not isinstance(text, str): return True
        stripped = text.strip()
        if not stripped: return True
        import string
        punct = set(string.punctuation)
        return all(c in punct | {" ", "\t", "\n", "\r", "\u00a0"} for c in stripped)

    validated = []
    all_invalid = []
    error_counts = Counter2()

    for rid, rec in candidates.items():
        text = rec["text"]
        out = existing.get(rid, {})
        raw_response = out.get("raw_response", "")
        api_error = out.get("error")

        if api_error:
            error_counts["api_error"] += 1
            all_invalid.append({
                "record_id": rid, "split": rec["split"],
                "reason": "api_error", "error_detail": api_error,
                "raw_response": raw_response[:300], "text": text,
            })
            validated.append({"id": rid, "split": rec["split"], "text": text, "spans": []})
            continue

        parsed = None
        if raw_response:
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                pass

        spans = []
        if isinstance(parsed, dict):
            spans = parsed.get("complaint_spans", [])
        if not isinstance(spans, list):
            all_invalid.append({
                "record_id": rid, "split": rec["split"],
                "reason": "complaint_spans_not_list", "text": text,
            })
            validated.append({"id": rid, "split": rec["split"], "text": text, "spans": []})
            continue

        clean_spans = []
        for idx, span in enumerate(spans):
            if not isinstance(span, dict):
                all_invalid.append({
                    "record_id": rid, "split": rec["split"],
                    "reason": "span_not_dict", "span_index": idx, "text": text,
                })
                continue
            span_text = span.get("text", "")
            if is_invalid_span_text(span_text):
                all_invalid.append({
                    "record_id": rid, "split": rec["split"],
                    "reason": "invalid_span_text", "span_index": idx,
                    "text": span_text[:100], "review_text": text,
                })
                continue
            offset = find_span_offset(text, span_text)
            if offset is None:
                all_invalid.append({
                    "record_id": rid, "split": rec["split"],
                    "reason": "text_not_in_review", "span_index": idx,
                    "text": span_text[:200], "review_text": text,
                })
                continue
            clean_spans.append({"text": span_text, "start": offset[0], "end": offset[1]})

        # Deduplicate
        seen = set()
        unique = []
        for s in clean_spans:
            if s["text"] not in seen:
                seen.add(s["text"])
                unique.append(s)
        clean_spans = unique

        # Resolve nesting (keep longer)
        if len(clean_spans) > 1:
            sorted_spans = sorted(clean_spans, key=lambda s: (s["end"] - s["start"], s["start"]))
            resolved = []
            for span in sorted_spans:
                dominated = any(
                    e["start"] <= span["start"] and span["end"] <= e["end"]
                    for e in resolved
                )
                if dominated:
                    continue
                resolved = [e for e in resolved
                           if not (span["start"] <= e["start"] and e["end"] <= span["end"])]
                resolved.append(span)
            clean_spans = resolved

        validated.append({"id": rid, "split": rec["split"], "text": text, "spans": clean_spans})

    # Save final outputs
    total_spans = sum(len(r["spans"]) for r in validated)
    records_with = sum(1 for r in validated if r["spans"])
    records_without = sum(1 for r in validated if not r["spans"])
    avg_spans = total_spans / len(validated) if validated else 0
    split_dist = Counter2(r["split"] for r in validated)

    valid_path = DATA_OUT / "pilot_validated_outputs.jsonl"
    with open(valid_path, "w", encoding="utf-8") as f:
        for rec in validated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {valid_path.name}")

    if all_invalid:
        invalid_path = DATA_OUT / "pilot_invalid_outputs.csv"
        fieldnames = ["record_id", "split", "reason", "span_index", "error_detail", "text"]
        with open(invalid_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_invalid)
        print(f"  Saved {invalid_path.name}: {len(all_invalid)} rows")

    manifest = {
        "pilot_records":             len(validated),
        "records_with_spans":       records_with,
        "records_without_spans":    records_without,
        "total_spans":            total_spans,
        "average_spans_per_record": round(avg_spans, 3),
        "invalid_output_count":     len(all_invalid),
        "invalid_span_text_count":  error_counts.get("invalid_span_text", 0),
        "json_parse_error_count":   error_counts.get("json_parse_error", 0),
        "text_not_found_count":     error_counts.get("text_not_in_review", 0),
        "api_error_count":         error_counts.get("api_error", 0),
        "split_distribution":       dict(split_dist),
        "model":                  GEMINI_MODEL,
    }

    manifest_path = DATA_OUT / "pilot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  Saved {manifest_path.name}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  1. Pilot records:                   {manifest['pilot_records']}")
    print(f"  2. Records with at least one span: {manifest['records_with_spans']}")
    print(f"  3. Total extracted spans:           {manifest['total_spans']}")
    print(f"  4. Average spans per record:        {manifest['average_spans_per_record']}")
    print(f"  5. Invalid JSON outputs:            {manifest['json_parse_error_count']}")
    print(f"  6. Span texts not in review:        {manifest['text_not_found_count']}")
    print(f"  7. API/rate-limit errors:          {manifest['api_error_count']}")
    print(f"\n  ✅ Retry complete.")


if __name__ == "__main__":
    main()
