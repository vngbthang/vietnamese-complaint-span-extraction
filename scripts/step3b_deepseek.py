#!/usr/bin/env python3
"""
Step 3B: Full ViOCD Complaint Span Annotation — DeepSeek Token-Index
====================================================================
Annotates all positive complaint records using token-index strategy
with deepseek-ai/deepseek-v4-flash via OpenAI client.
Checkpoint/resume enabled for interruption recovery.
API key read from environment only.
"""

import csv
import json
import os
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_IN   = REPO_ROOT / "data" / "complaint_span_annotations"
NEG_IN    = REPO_ROOT / "data" / "bio_splits" / "complaint_detection"
DATA_OUT  = REPO_ROOT / "data" / "complaint_span_annotations" / "full_token_index_deepseek"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# API config — read from environment only
# ─────────────────────────────────────────────────────────
NVIDIA_API_KEY  = os.environ["NVIDIA_API_KEY"]
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL           = os.environ.get("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash")
API_TIMEOUT     = 120
MAX_RETRIES    = 5
RETRY_DELAYS   = [2, 4, 8, 16, 32]

# ─────────────────────────────────────────────────────────
# OpenAI client
# ─────────────────────────────────────────────────────────
client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
    timeout=API_TIMEOUT,
)

# ─────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────

def tokenize(text: str) -> List[Dict]:
    tokens = []
    for m in re.finditer(r"\S+", text):
        tokens.append({
            "idx":   len(tokens),
            "text":  m.group(),
            "start": m.start(),
            "end":   m.end(),
        })
    return tokens


# ─────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are annotating Vietnamese customer reviews for complaint span extraction.

A complaint span is a contiguous segment of the review that directly expresses dissatisfaction, product/service failure, inconvenience, unmet expectation, or negative customer experience.

You will receive the original review and a list of indexed tokens.
Your task is to select token index ranges that form complaint spans.

Annotation rules:
1. Return token index ranges only.
2. Do not return text spans.
3. Do not return character offsets.
4. Each complaint span must be a contiguous token range from the provided token list.
5. Do not extract aspect terms alone.
6. Prefer minimal but complete complaint expressions.
7. Multiple complaint spans are allowed.
8. Return an empty list if there is no explicit complaint expression.
9. Do not annotate suggestions or feature requests unless they clearly express dissatisfaction or a product/service failure.
10. If unsure, return an empty list."""


def build_user_prompt(record_id: str, text: str, tokens: List[Dict]) -> str:
    token_lines = "\n".join(f"[{t['idx']}] {t['text']}" for t in tokens)
    return (
        f"Review ID:\n{record_id}\n\n"
        f"Original review:\n<review>\n{text}\n</review>\n\n"
        f"Indexed tokens:\n{token_lines}\n\n"
        f'Return only valid JSON:\n{{\n  "id": "{record_id}",\n'
        f'  "complaint_token_spans": [\n    {{\n      "start_token": 0,\n      "end_token": 2\n    }}\n  ]\n}}'
    )


# ─────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────

def call_api_with_retry(messages: List[Dict], max_retries: int = MAX_RETRIES) -> Tuple[Optional[str], Optional[str]]:
    """Call DeepSeek API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
                stream=False,
            )
            content = completion.choices[0].message.content
            return content, None
        except Exception as e:
            err_str = str(e)
            # Rate limit / transient errors
            if attempt < max_retries - 1:
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"    [retry {attempt+1}/{max_retries}] {err_str[:80]} — waiting {wait}s")
                time.sleep(wait)
                continue
            return None, err_str
    return None, "max_retries_exceeded"


def extract_json(content: str) -> Optional[dict]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    code_lines, in_code = [], False
    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)

    if code_lines:
        try:
            return json.loads("\n".join(code_lines).strip())
        except json.JSONDecodeError:
            pass

    for start_char, start_idx in [("[", content.find("[")), ("{", content.find("{"))]:
        if start_idx == -1:
            continue
        bracket, close = start_char, "]" if start_char == "[" else "}"
        depth, end_idx, in_string = 0, start_idx, False
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
# Token span parser (handles both [s,e] and {start_token,end_token})
# ─────────────────────────────────────────────────────────

def safe_to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_token_span(item) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if isinstance(item, dict):
        s = item.get("start_token")
        e = item.get("end_token")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        s = safe_to_int(item[0])
        e = safe_to_int(item[1])
    else:
        return None, None, "invalid_span_format"

    if s is None or e is None:
        return None, None, "token_not_integer"
    if not isinstance(s, int) or not isinstance(e, int):
        return None, None, "token_not_integer"
    if s < 0 or e < 0:
        return None, None, "negative_index"

    return s, e, None


# ─────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────

def validate_record(
    record_id: str,
    review_text: str,
    tokens: List[Dict],
    raw_response: str,
) -> Tuple[List[Dict], List[Dict], Counter]:
    stats = Counter()
    invalid = []
    accepted = []
    n_tokens = len(tokens)
    offsets = {t["idx"]: t for t in tokens}

    parsed = extract_json(raw_response)
    if parsed is None:
        stats["json_invalid"] += 1
        invalid.append({"id": record_id, "review_text": review_text,
                      "predicted_value": raw_response[:200], "reason": "json_invalid"})
        return [], invalid, stats

    if isinstance(parsed, dict):
        token_spans = parsed.get("complaint_token_spans", [])
    elif isinstance(parsed, list) and parsed:
        token_spans = parsed[0].get("complaint_token_spans", []) if isinstance(parsed[0], dict) else []
    else:
        token_spans = None

    if not isinstance(token_spans, list):
        stats["not_list"] += 1
        invalid.append({"id": record_id, "review_text": review_text,
                      "predicted_value": str(token_spans)[:100], "reason": "complaint_spans_not_list"})
        return [], invalid, stats

    if len(token_spans) == 0:
        return [], invalid, stats

    for idx, item in enumerate(token_spans):
        s, e, err = parse_token_span(item)
        if err:
            stats[err] += 1
            invalid.append({"id": record_id, "review_text": review_text,
                          "predicted_value": str(item)[:100], "reason": err})
            continue

        if s > e:
            stats["start_gt_end"] += 1
            invalid.append({"id": record_id, "review_text": review_text,
                          "predicted_value": f"[{s}, {e}]", "reason": "start_token_gt_end_token"})
            continue

        if not (0 <= s <= e < n_tokens):
            stats["out_of_range"] += 1
            invalid.append({"id": record_id, "review_text": review_text,
                          "predicted_value": f"s={s}, e={e}, n={n_tokens}", "reason": "token_out_of_range"})
            continue

        char_start = offsets[s]["start"]
        char_end   = offsets[e]["end"]
        span_text  = review_text[char_start:char_end]

        accepted.append({
            "start":           char_start,
            "end":             char_end,
            "text":            span_text,
            "start_token":     s,
            "end_token":       e,
            "span_type":       "complaint",
            "offset_source":   "token_index_range",
            "annotation_model": MODEL,
            "annotation_mode": "llm_token_index",
        })

    return accepted, invalid, stats


# ─────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────

def deduplicate(spans: List[Dict]) -> List[Dict]:
    seen, unique = set(), []
    for s in spans:
        if s["text"] not in seen:
            seen.add(s["text"]); unique.append(s)
    return unique


def resolve_nested(spans: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    if len(spans) <= 1:
        return spans, []
    sorted_spans = sorted(spans, key=lambda s: (s["end"] - s["start"], s["start"]))
    kept, removed = [], []
    for span in sorted_spans:
        if any(e["start"] <= span["start"] and span["end"] <= e["end"] for e in kept):
            removed.append({**span, "reason": "nested_span_removed"})
            continue
        kept = [e for e in kept if not (span["start"] <= e["start"] and e["end"] <= span["end"])]
        kept.append(span)
    return kept, removed


# ─────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────

def load_positive_records() -> Dict[str, List[Dict]]:
    files = {
        "train": DATA_IN / "viocd_complaint_candidates_train.jsonl",
        "valid": DATA_IN / "viocd_complaint_candidates_valid.jsonl",
        "test":  DATA_IN / "viocd_complaint_candidates_test.jsonl",
    }
    records = {}
    for split, path in files.items():
        records[split] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["_is_positive"] = True
                records[split].append(rec)
    return records


def load_negative_records() -> Dict[str, List[Dict]]:
    files = {
        "train": NEG_IN / "train.jsonl",
        "valid": NEG_IN / "valid.jsonl",
        "test":  NEG_IN / "test.jsonl",
    }
    records = {}
    for split, path in files.items():
        records[split] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("label") == 0:
                    rec["_is_positive"] = False
                    records[split].append(rec)
    return records


# ─────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────

def load_checkpoint() -> Dict[str, Dict]:
    ckpt = DATA_OUT / "full_raw_outputs.jsonl"
    if not ckpt.exists():
        return {}
    results = {}
    with open(ckpt, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            results[obj["id"]] = obj
    return results


def save_checkpoint(result: Dict):
    ckpt = DATA_OUT / "full_raw_outputs.jsonl"
    with open(ckpt, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()


# ─────────────────────────────────────────────────────────
# Main annotation
# ─────────────────────────────────────────────────────────

def run_annotation(positive_records: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    all_records = []
    for split, recs in positive_records.items():
        for rec in recs:
            rec["_split"] = split
            all_records.append(rec)

    total = len(all_records)
    print(f"\n  Total positive records: {total}")

    checkpoint = load_checkpoint()
    annotated = len(checkpoint)
    print(f"  Checkpoint: {annotated}/{total} already annotated")

    to_annotate = [r for r in all_records if r["id"] not in checkpoint]

    if not to_annotate:
        print(f"  All {total} already annotated — resuming validation only")
        return checkpoint

    print(f"  To annotate: {len(to_annotate)}")

    for i, rec in enumerate(to_annotate):
        rid = rec["id"]
        tokens = tokenize(rec["text"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(rid, rec["text"], tokens)},
        ]

        content, error = call_api_with_retry(messages)
        result = {
            "id":           rid,
            "raw_response": content if content else "",
            "error":        error,
            "n_tokens":     len(tokens),
            "split":        rec["_split"],
        }
        checkpoint[rid] = result
        save_checkpoint(result)
        annotated += 1

        if i == 0 or (i + 1) % 25 == 0 or i == len(to_annotate) - 1:
            pct = annotated / total * 100
            print(f"  Progress: {annotated}/{total} ({pct:.1f}%) — last: {rid}")

        time.sleep(0.1)

    return checkpoint


# ─────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────

def validate_all(
    positive_records: Dict[str, List[Dict]],
    raw_outputs: Dict[str, Dict],
) -> Tuple[List[Dict], List[Dict], Counter]:
    validated = []
    all_invalid = []
    agg = Counter()

    for split, recs in positive_records.items():
        for rec in recs:
            rid = rec["id"]
            text = rec["text"]
            tokens = tokenize(text)
            out = raw_outputs.get(rid, {})
            raw = out.get("raw_response", "")
            error = out.get("error")

            if not raw and error:
                agg["missing_output"] += 1
                all_invalid.append({"id": rid, "review_text": text,
                                   "predicted_value": error, "reason": "missing_output"})
                validated.append({"id": rid, "split": rec["_split"], "text": text,
                                "n_tokens": len(tokens), "spans": []})
                continue

            accepted, invalid, stats = validate_record(rid, text, tokens, raw)
            for k, v in stats.items():
                agg[k] += v
            all_invalid.extend(invalid)

            accepted = deduplicate(accepted)
            accepted, removed = resolve_nested(accepted)
            for r in removed:
                all_invalid.append({
                    "id": rid, "review_text": text,
                    "predicted_value": f"[{r.get('start_token')}, {r.get('end_token')}]",
                    "reason": r["reason"],
                })

            validated.append({"id": rid, "split": rec["_split"], "text": text,
                            "n_tokens": len(tokens), "spans": accepted})

    return validated, all_invalid, agg


# ─────────────────────────────────────────────────────────
# Add negatives and save
# ─────────────────────────────────────────────────────────

def add_negative_records(
    validated_positive: List[Dict],
    negative_records: Dict[str, List[Dict]],
) -> List[Dict]:
    combined = list(validated_positive)
    for split, recs in negative_records.items():
        for rec in recs:
            combined.append({
                "id":           rec["id"],
                "split":        rec.get("split", split),
                "source":       rec.get("source", "bio_splits"),
                "text":         rec["text"],
                "n_tokens":     len(tokenize(rec["text"])),
                "spans":        [],
                "_is_negative": True,
            })
    return combined


def save_final_splits(validated_positive: List[Dict], combined: List[Dict]):
    for split in ["train", "valid", "test"]:
        pos = [r for r in validated_positive if r["split"] == split]
        neg = [r for r in combined if r.get("_is_negative") and r["split"] == split]

        out_records = []
        for r in pos:
            out_records.append({
                "id":                 r["id"],
                "source":             "viocd",
                "task":              "complaint_span_extraction",
                "split":             r["split"],
                "text":              r["text"],
                "review_level_label": 1,
                "spans":             r["spans"],
            })
        for r in neg:
            out_records.append({
                "id":                 r["id"],
                "source":             r.get("source", "bio_splits"),
                "task":              "complaint_span_extraction",
                "split":             r["split"],
                "text":              r["text"],
                "review_level_label": 0,
                "spans":             [],
            })

        out_path = DATA_OUT / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split}.jsonl: {len(out_records)} records "
              f"({len(pos)} positive + {len(neg)} negative)")


def build_manifest(
    validated_positive: List[Dict],
    combined: List[Dict],
    invalid: List[Dict],
    agg: Counter,
) -> Dict:
    pos_records  = validated_positive
    neg_records  = [r for r in combined if r.get("_is_negative")]
    pos_with     = sum(1 for r in pos_records if r["spans"])
    pos_without  = sum(1 for r in pos_records if not r["spans"])
    pos_spans    = sum(len(r["spans"]) for r in pos_records)

    return {
        "annotation_model":                    MODEL,
        "total_records":                    len(combined),
        "positive_review_records":          len(pos_records),
        "negative_all_o_records":          len(neg_records),
        "records_with_spans":             pos_with,
        "records_without_spans":       pos_without + len(neg_records),
        "total_complaint_spans":      pos_spans,
        "average_spans_per_positive_record": round(pos_spans / len(pos_records), 3) if pos_records else 0,
        "invalid_json_outputs":          agg["json_invalid"],
        "invalid_token_range_count":    (agg["out_of_range"] + agg["not_integer"]
                                        + agg["start_gt_end"] + agg["negative_index"]
                                        + agg["invalid_span_format"] + agg["missing_output"]),
        "out_of_range_count":          agg["out_of_range"],
        "token_not_integer_count":    agg["not_integer"],
        "missing_output_records":       agg["missing_output"],
        "nested_span_removed_count":  sum(1 for e in invalid if e.get("reason") == "nested_span_removed"),
        "splits": {
            split: {
                "records":                sum(1 for r in combined if r["split"] == split),
                "positive_records":       sum(1 for r in pos_records if r["split"] == split),
                "negative_records":       sum(1 for r in neg_records if r["split"] == split),
                "records_with_spans":    sum(1 for r in pos_records if r["split"] == split and r["spans"]),
                "records_without_spans": sum(1 for r in pos_records if r["split"] == split and not r["spans"])
                                        + sum(1 for r in neg_records if r["split"] == split),
                "total_spans":          sum(len(r["spans"]) for r in pos_records if r["split"] == split),
            }
            for split in ["train", "valid", "test"]
        },
    }


def save_examples(validated_positive: List[Dict], n: int = 20):
    examples_path = DATA_OUT / "full_annotation_examples.txt"
    shown = 0
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Step 3B: Full Annotation Examples — DeepSeek Token-Index\n")
        f.write("=" * 70 + "\n\n")
        for rec in validated_positive:
            if shown >= n or not rec["spans"]:
                continue
            shown += 1
            f.write(f"[{shown}] ID: {rec['id']}  ({rec['split']})\n")
            f.write(f"    Review ({rec['n_tokens']} tokens): {repr(rec['text'][:80])}\n")
            f.write(f"    Spans ({len(rec['spans'])}):\n")
            for i, span in enumerate(rec["spans"][:5]):
                f.write(f"      [{i+1}] tokens[{span['start_token']}:{span['end_token']}] "
                       f"chars[{span['start']},{span['end']}] {repr(span['text'][:60])}\n")
            f.write("\n")
        f.write(f"Total shown: {shown}\n")
    print(f"\n  Saved examples to {examples_path.name}")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("Step 3B: Full ViOCD Annotation — DeepSeek Token-Index")
    print("=" * 70)
    print(f"\n  Output:  {DATA_OUT}")
    print(f"  Model:   {MODEL}")
    print(f"  Client:  {NVIDIA_BASE_URL}\n")

    # 1. Load
    print("-" * 70)
    print("1. LOADING DATA")
    print("-" * 70)
    pos_records = load_positive_records()
    neg_records = load_negative_records()
    pos_total = sum(len(v) for v in pos_records.values())
    neg_total = sum(len(v) for v in neg_records.values())
    print(f"  Positive records:  {pos_total} "
          f"(train={len(pos_records['train'])}, "
          f"valid={len(pos_records['valid'])}, "
          f"test={len(pos_records['test'])})")
    print(f"  Negative records: {neg_total} "
          f"(train={len(neg_records['train'])}, "
          f"valid={len(neg_records['valid'])}, "
          f"test={len(neg_records['test'])})")

    # 2. Annotation
    print("\n" + "-" * 70)
    print("2. RUNNING ANNOTATION (checkpoint/resume)")
    print("-" * 70)
    raw_outputs = run_annotation(pos_records)
    print(f"\n  Total annotated: {len(raw_outputs)}")

    # 3. Validation
    print("\n" + "-" * 70)
    print("3. VALIDATING AND POST-PROCESSING")
    print("-" * 70)
    validated_pos, invalid, agg = validate_all(pos_records, raw_outputs)
    pos_with  = sum(1 for r in validated_pos if r["spans"])
    pos_spans = sum(len(r["spans"]) for r in validated_pos)
    print(f"  Validated:           {len(validated_pos)}")
    print(f"  With spans:         {pos_with}")
    print(f"  Total valid spans: {pos_spans}")
    print(f"  Invalid entries:     {len(invalid)}")
    print(f"    json_invalid:    {agg['json_invalid']}")
    print(f"    out_of_range:   {agg['out_of_range']}")
    print(f"    not_integer:    {agg['not_integer']}")
    print(f"    missing_output: {agg['missing_output']}")

    # 4. Add negatives
    print("\n" + "-" * 70)
    print("4. ADDING NEGATIVE ALL-O RECORDS")
    print("-" * 70)
    combined = add_negative_records(validated_pos, neg_records)
    print(f"  Combined: {len(combined)} "
          f"({len(validated_pos)} positive + {len(combined)-len(validated_pos)} negative)")

    # 5. Save
    print("\n" + "-" * 70)
    print("5. SAVING OUTPUTS")
    print("-" * 70)

    with open(DATA_OUT / "full_validated_positive_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in validated_pos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  full_validated_positive_outputs.jsonl: {len(validated_pos)} records")

    if invalid:
        with open(DATA_OUT / "full_invalid_outputs.csv", "w", newline="", encoding="utf-8") as f:
            fn = ["id", "reason", "predicted_value", "review_text"]
            w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
            w.writeheader(); w.writerows(invalid)
        print(f"  full_invalid_outputs.csv: {len(invalid)} rows")

    save_final_splits(validated_pos, combined)

    manifest = build_manifest(validated_pos, combined, invalid, agg)
    with open(DATA_OUT / "full_annotation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  full_annotation_manifest.json saved")

    save_examples(validated_pos, n=20)

    # Summary
    elapsed = time.time() - t0
    neg_count = len(combined) - len(validated_pos)
    inv_count = manifest["invalid_token_range_count"] + manifest["invalid_json_outputs"]

    print("\n" + "=" * 70)
    print("MANIFEST SUMMARY")
    print("=" * 70)
    print(f"\n  1. Positive records annotated:  {len(validated_pos)}")
    print(f"  2. Negative all-O added:       {neg_count}")
    print(f"  3. Total valid spans:         {pos_spans}")
    print(f"  4. Records with spans:        {pos_with}")
    print(f"  5. Records without spans:     {len(combined) - pos_with}")
    print(f"  6. Invalid JSON outputs:      {manifest['invalid_json_outputs']}")
    print(f"  7. Invalid token range count: {manifest['invalid_token_range_count']}")
    print(f"  8. Out of range count:       {manifest['out_of_range_count']}")
    print(f"\n  Split distribution:")
    for split, s in manifest["splits"].items():
        print(f"    {split}: {s['records']} records "
              f"({s['positive_records']} pos + {s['negative_records']} neg), "
              f"{s['records_with_spans']} with spans, {s['total_spans']} spans")

    print(f"\n  [Output files in {DATA_OUT}]")
    for f in sorted(DATA_OUT.iterdir()):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"\n  ✅ Step 3B complete.")


if __name__ == "__main__":
    main()
