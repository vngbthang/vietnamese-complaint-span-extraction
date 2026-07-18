#!/usr/bin/env python3
"""
Step 3A: Pilot LLM Annotation for ViOCD Complaint Spans
======================================================
Uses NVIDIA/Mistral API via direct HTTP (no SDK needed).

Key design: LLM returns only span text strings (no offsets).
Script computes start/end via exact string matching.

Pilot: 80 train + 20 valid (seed=42)
"""

import csv
import json
import random
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_IN    = REPO_ROOT / "data" / "complaint_span_annotations"
PILOT_DIR  = DATA_IN  / "pilot"
DATA_OUT   = PILOT_DIR

PILOT_TRAIN_SIZE = 80
PILOT_VALID_SIZE = 20
RANDOM_SEED = 42

# NVIDIA API config
NVIDIA_API_KEY = "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"
NVIDIA_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL          = "mistralai/mistral-small-4-119b-2603"
BATCH_SIZE     = 1   # one record at a time for reliable JSON parsing
API_DELAY      = 0.3  # seconds between calls

# ─────────────────────────────────────────────────────────
# Annotation prompt
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là người gán nhãn đoạn phàn nàn trong đánh giá tiếng Việt.

Nhiệm vụ:
Trích xuất tất cả các đoạn phàn nàn từ đánh giá.

Định nghĩa:
Đoạn phàn nàn là đoạn văn bản liên tục thể hiện sự không hài lòng, lỗi sản phẩm/dịch vụ, bất tiện, kỳ vọng không đáp ứng, hoặc trải nghiệm tiêu cực của khách hàng.

Quy tắc:
1. Chỉ trích xuất chính xác chuỗi con từ đánh giá gốc.
2. Không viết lại, chuẩn hóa, dịch, hay diễn giải.
3. Không suy luận phàn nàn ẩn.
4. Không trích xuất thuật ngữ khía cạnh đơn lẻ.
   Bad: "pin"
   Good: "pin tụt quá nhanh"
5. Ưu tiên các đoạn nhỏ nhưng đầy đủ.
6. Cho phép nhiều đoạn phàn nàn.
7. Trả về danh sách rỗng nếu không có phàn nàn rõ ràng.
8. Mỗi đoạn trả về phải xuất hiện chính xác trong đánh giá gốc.
9. Chỉ trả về text — KHÔNG có offset ký tự.

Trả về CHỈ JSON array, không giải thích, không markdown:
[
  {"id": "...", "complaint_spans": [{"text": "..."}]},
  ...
]"""


# ─────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────

def call_nvidia(record: Dict) -> Dict:
    """Call NVIDIA API for a single record."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f'ID: {record["id"]}\nText: {record["text"]}\n'},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(
            NVIDIA_URL,
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
            return {"id": record["id"], "raw_response": "", "error": msg}

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Parse JSON from response
        # Remove markdown code fences
        cleaned = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*$",   "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try wrapping if it's a single object
            try:
                parsed = json.loads(content.strip())
            except json.JSONDecodeError:
                return {"id": record["id"], "raw_response": content[:1000], "error": f"json_parse_error"}

        # Extract complaint_spans
        if isinstance(parsed, dict) and "complaint_spans" in parsed:
            return {
                "id":            record["id"],
                "raw_response":  json.dumps(parsed, ensure_ascii=False),
                "error":         None,
            }
        elif isinstance(parsed, list):
            # Find the entry matching this record's id
            for item in parsed:
                if isinstance(item, dict) and item.get("id") == record["id"]:
                    return {
                        "id":           record["id"],
                        "raw_response": json.dumps(item, ensure_ascii=False),
                        "error":        None,
                    }
            return {"id": record["id"], "raw_response": json.dumps(parsed, ensure_ascii=False), "error": "id_not_found"}
        else:
            return {"id": record["id"], "raw_response": content[:500], "error": "unexpected_structure"}

    except requests.exceptions.Timeout:
        return {"id": record["id"], "raw_response": "", "error": "timeout"}
    except Exception as e:
        return {"id": record["id"], "raw_response": "", "error": str(e)}


# ─────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────

def find_span_offset(text: str, span_text: str) -> Optional[Tuple[int, int]]:
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
    if not text or not isinstance(text, str):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    import string
    punct = set(string.punctuation)
    return all(c in punct | {" ", "\t", "\n", "\r", "\u00a0"} for c in stripped)


def parse_and_validate(record: Dict, raw_response: str) -> Tuple[List[Dict], List[Dict]]:
    rid  = record["id"]
    text = record["text"]
    validated = []
    invalid   = []

    parsed = None
    if raw_response:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            invalid.append({
                "record_id": rid, "reason": "json_parse_error",
                "raw_response": raw_response[:300], "text": text,
            })
            return [], invalid

    if not isinstance(parsed, dict):
        invalid.append({
            "record_id": rid, "reason": "json_not_dict",
            "raw_response": raw_response[:300], "text": text,
        })
        return [], invalid

    spans = parsed.get("complaint_spans", [])
    if not isinstance(spans, list):
        invalid.append({
            "record_id": rid, "reason": "complaint_spans_not_list",
            "raw_response": raw_response[:300], "text": text,
        })
        return [], invalid

    for idx, span in enumerate(spans):
        if not isinstance(span, dict):
            invalid.append({
                "record_id": rid, "reason": "span_not_dict",
                "span_index": idx, "raw_response": raw_response[:300],
                "text": str(span)[:100],
            })
            continue

        span_text = span.get("text", "")

        if is_invalid_span_text(span_text):
            invalid.append({
                "record_id": rid, "reason": "invalid_span_text",
                "span_index": idx, "raw_response": raw_response[:300],
                "text": span_text[:100], "review_text": text,
            })
            continue

        offset = find_span_offset(text, span_text)
        if offset is None:
            invalid.append({
                "record_id": rid, "reason": "text_not_in_review",
                "span_index": idx, "raw_response": raw_response[:300],
                "text": span_text[:200], "review_text": text,
            })
            continue

        validated.append({"text": span_text, "start": offset[0], "end": offset[1]})

    return validated, invalid


def remove_exact_duplicates(spans: List[Dict]) -> List[Dict]:
    seen, unique = set(), []
    for s in spans:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    return unique


def resolve_nested_spans(spans: List[Dict]) -> List[Dict]:
    if len(spans) <= 1:
        return spans
    sorted_spans = sorted(spans, key=lambda s: (s["end"] - s["start"], s["start"]))
    resolved = []
    for span in sorted_spans:
        if any(e["start"] <= span["start"] and span["end"] <= e["end"] for e in resolved):
            continue
        resolved = [e for e in resolved
                   if not (span["start"] <= e["start"] and e["end"] <= span["end"])]
        resolved.append(span)
    return resolved


# ─────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────

def load_candidates() -> Tuple[List[Dict], List[Dict]]:
    train_path = DATA_IN / "viocd_complaint_candidates_train.jsonl"
    valid_path = DATA_IN / "viocd_complaint_candidates_valid.jsonl"
    train_recs, valid_recs = [], []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["split"] = "train"
            train_recs.append(rec)
    with open(valid_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["split"] = "valid"
            valid_recs.append(rec)
    return train_recs, valid_recs


def sample_pilot(train_recs: List[Dict], valid_recs: List[Dict]) -> List[Dict]:
    random.seed(RANDOM_SEED)
    return (
        random.sample(train_recs, min(PILOT_TRAIN_SIZE, len(train_recs))) +
        random.sample(valid_recs, min(PILOT_VALID_SIZE, len(valid_recs)))
    )


def run_annotation(pilot: List[Dict]) -> List[Dict]:
    print(f"  Model: {MODEL}")
    print(f"  Batch size: {BATCH_SIZE} (one record per call)")
    print(f"  Total: {len(pilot)} records")

    all_results = []
    for i, rec in enumerate(pilot):
        if i == 0 or (i + 1) % 10 == 0 or i == len(pilot) - 1:
            print(f"  Annotating {i+1}/{len(pilot)}...")

        result = call_nvidia(rec)
        all_results.append(result)

        # Save interim every 25 records
        if (i + 1) % 25 == 0:
            with open(DATA_OUT / "pilot_raw_outputs.jsonl", "w", encoding="utf-8") as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"    [Saved interim: {len(all_results)} results]")

        time.sleep(API_DELAY)

    return all_results


def validate_all(pilot: List[Dict], raw_outputs: List[Dict]) -> Tuple[List[Dict], List[Dict], Counter]:
    rec_by_id = {rec["id"]: rec for rec in pilot}
    out_by_id = {out["id"]:  out  for out  in raw_outputs}

    validated_records = []
    all_invalid       = []
    error_counts      = Counter()

    for rec in pilot:
        rid   = rec["id"]
        text  = rec["text"]
        out   = out_by_id.get(rid, {})
        raw   = out.get("raw_response", "")
        error = out.get("error")

        if error:
            error_counts["api_error"] += 1
            all_invalid.append({
                "record_id": rid, "split": rec["split"],
                "reason": "api_error", "error_detail": error,
                "raw_response": raw[:300], "text": text,
            })
            validated_records.append({"id": rid, "split": rec["split"], "text": text, "spans": []})
            continue

        spans, invalid = parse_and_validate(rec, raw)
        for entry in invalid:
            entry["split"] = rec["split"]
            entry["review_text"] = text
            all_invalid.append(entry)

        spans = remove_exact_duplicates(spans)
        spans = resolve_nested_spans(spans)

        validated_records.append({"id": rid, "split": rec["split"], "text": text, "spans": spans})

    return validated_records, all_invalid, error_counts


def save_outputs(raw_outputs, validated, invalid, error_counts):
    # Raw
    with open(DATA_OUT / "pilot_raw_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in raw_outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  pilot_raw_outputs.jsonl: {len(raw_outputs)} records")

    # Validated
    with open(DATA_OUT / "pilot_validated_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in validated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  pilot_validated_outputs.jsonl: {len(validated)} records")

    # Invalid CSV
    if invalid:
        with open(DATA_OUT / "pilot_invalid_outputs.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["record_id", "split", "reason", "span_index", "error_detail", "text", "review_text"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(invalid)
        print(f"  pilot_invalid_outputs.csv: {len(invalid)} rows")

    # Manifest
    total_spans       = sum(len(r["spans"]) for r in validated)
    records_with      = sum(1 for r in validated if r["spans"])
    records_without   = sum(1 for r in validated if not r["spans"])
    avg_spans         = total_spans / len(validated) if validated else 0
    split_dist        = dict(Counter(r["split"] for r in validated))

    manifest = {
        "pilot_records":             len(validated),
        "records_with_spans":       records_with,
        "records_without_spans":    records_without,
        "total_spans":            total_spans,
        "average_spans_per_record": round(avg_spans, 3),
        "invalid_output_count":     len(invalid),
        "invalid_span_text_count":  error_counts.get("invalid_span_text", 0),
        "json_parse_error_count":   error_counts.get("json_parse_error", 0),
        "text_not_found_count":     error_counts.get("text_not_in_review", 0),
        "api_error_count":         error_counts.get("api_error", 0),
        "split_distribution":       split_dist,
        "model":                  MODEL,
    }

    with open(DATA_OUT / "pilot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  pilot_manifest.json saved")

    return manifest


def print_examples(validated: List[Dict], n: int = 10):
    shown = 0
    for rec in validated:
        if shown >= n:
            break
        spans = rec["spans"]
        if not spans:
            continue
        shown += 1
        print(f"\n  [{shown}] ID: {rec['id']}  ({rec['split']})")
        print(f"      Text: {repr(rec['text'][:100])}")
        print(f"      Spans ({len(spans)}):")
        for i, span in enumerate(spans[:5]):
            print(f"        [{i+1}] [{span['start']},{span['end']}] {repr(span['text'][:60])}")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Step 3A: Pilot LLM Annotation (NVIDIA/Mistral) — ViOCD")
    print("=" * 70)
    print(f"\n  Input:   {DATA_IN}")
    print(f"  Output:  {DATA_OUT}")
    print(f"  Model:   {MODEL}")
    print(f"  Pilot:   {PILOT_TRAIN_SIZE} train + {PILOT_VALID_SIZE} valid\n")

    # 1. Load and sample
    print("-" * 70)
    print("1. LOADING AND SAMPLING")
    train_recs, valid_recs = load_candidates()
    pilot = sample_pilot(train_recs, valid_recs)
    dist  = Counter(r["split"] for r in pilot)
    print(f"  Train: {dist.get('train', 0)}, Valid: {dist.get('valid', 0)}, Total: {len(pilot)}")

    # 2. Annotate
    print("\n" + "-" * 70)
    print("2. RUNNING ANNOTATION")
    raw_outputs = run_annotation(pilot)

    # 3. Validate
    print("\n" + "-" * 70)
    print("3. VALIDATING")
    validated, invalid, error_counts = validate_all(pilot, raw_outputs)
    print(f"  With spans:     {sum(1 for r in validated if r['spans'])}")
    print(f"  Without spans:  {sum(1 for r in validated if not r['spans'])}")
    print(f"  Total spans:    {sum(len(r['spans']) for r in validated)}")
    print(f"  API errors:     {error_counts.get('api_error', 0)}")
    print(f"  JSON parse err: {error_counts.get('json_parse_error', 0)}")
    print(f"  Text not found: {error_counts.get('text_not_in_review', 0)}")

    # 4. Save
    print("\n" + "-" * 70)
    print("4. SAVING OUTPUTS")
    manifest = save_outputs(raw_outputs, validated, invalid, error_counts)

    # 5. Examples
    print_examples(validated, n=10)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  1. Pilot records:                   {manifest['pilot_records']}")
    print(f"  2. Records with at least one span: {manifest['records_with_spans']}")
    print(f"  3. Total extracted spans:           {manifest['total_spans']}")
    print(f"  4. Average spans per record:        {manifest['average_spans_per_record']}")
    print(f"  5. Invalid JSON outputs:            {manifest['json_parse_error_count']}")
    print(f"  6. Span texts not in review:        {manifest['text_not_found_count']}")
    print(f"  7. API errors:                      {manifest['api_error_count']}")

    print(f"\n  [Output files in {DATA_OUT}]")
    for f in sorted(DATA_OUT.iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".csv", ".json"):
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  ✅ Step 3A complete.")


if __name__ == "__main__":
    main()
