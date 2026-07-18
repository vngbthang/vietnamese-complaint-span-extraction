#!/usr/bin/env python3
"""
Step 3A.2: Token-Index Annotation for ViOCD Complaint Spans
=========================================================
Strategy: Instead of copying span text, show indexed tokens to LLM,
ask for token index ranges, reconstruct spans from original using offsets.

Output: data/complaint_span_annotations/pilot_v3_token_index/
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
DATA_OUT  = DATA_IN  / "pilot_v3_token_index"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# API config
# ─────────────────────────────────────────────────────────
NVIDIA_API_KEY  = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL           = "mistralai/mistral-small-4-119b-2603"
API_DELAY       = 0.3

# ─────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────

def tokenize(text: str) -> List[Dict]:
    """
    Whitespace-aware tokenizer using pattern r"\\S+".
    Returns list of tokens with index and character offsets.
    """
    tokens = []
    for m in re.finditer(r"\S+", text):
        tokens.append({
            "idx":   len(tokens),
            "text":  m.group(),
            "start": m.start(),
            "end":   m.end(),
        })
    return tokens


def tokens_to_display(tokens: List[Dict]) -> str:
    """Format tokens as displayable indexed list."""
    lines = []
    for tok in tokens:
        # Escape double quotes in token text for display
        display_text = tok["text"].replace('"', '\\"')
        lines.append(f'[{tok["idx"]}] {display_text}')
    return "\n".join(lines)


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
10. If unsure, return an empty list.

Return JSON only."""


def build_user_prompt(record_id: str, text: str, tokens: List[Dict]) -> str:
    token_list = tokens_to_display(tokens)
    return f"""Review ID:
{record_id}

Original review:
<review>
{text}
</review>

Indexed tokens:
{token_list}

Return exactly:
{{
  "id": "{record_id}",
  "complaint_token_spans": [
    {{
      "start_token": 0,
      "end_token": 2
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────

def call_api(messages: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """Call NVIDIA API. Returns (content, error)."""
    payload = {
        "model": MODEL,
        "messages": messages,
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
            return None, err.get("message", f"HTTP {resp.status_code}")
        return resp.json()["choices"][0]["message"]["content"], None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


def extract_json(content: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    content = content.strip()

    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code fences
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

    # Bracket matching fallback
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
# Validation
# ─────────────────────────────────────────────────────────

def validate_record(
    record_id: str,
    review_text: str,
    tokens: List[Dict],
    raw_response: str,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Validate token-index output and reconstruct spans.
    Returns (accepted_spans, invalid_entries, stats).
    """
    stats = {
        "json_invalid": 0,
        "not_list": 0,
        "out_of_range": 0,
        "not_integer": 0,
        "start_gt_end": 0,
        "missing_field": 0,
    }
    invalid = []
    accepted = []
    n_tokens = len(tokens)
    token_offsets = {tok["idx"]: tok for tok in tokens}

    parsed = extract_json(raw_response)

    if parsed is None:
        stats["json_invalid"] += 1
        invalid.append({
            "id": record_id,
            "review_text": review_text,
            "predicted_value": raw_response[:200],
            "reason": "json_invalid",
        })
        return [], invalid, stats

    # Extract complaint_token_spans
    if isinstance(parsed, dict):
        token_spans = parsed.get("complaint_token_spans", [])
    elif isinstance(parsed, list) and parsed:
        token_spans = parsed[0].get("complaint_token_spans", []) if isinstance(parsed[0], dict) else []
    else:
        token_spans = None

    if not isinstance(token_spans, list):
        stats["not_list"] += 1
        invalid.append({
            "id": record_id,
            "review_text": review_text,
            "predicted_value": str(token_spans)[:100],
            "reason": "complaint_token_spans_not_list",
        })
        return [], invalid, stats

    if len(token_spans) == 0:
        return [], invalid, stats

    for idx, item in enumerate(token_spans):
        if not isinstance(item, dict):
            stats["missing_field"] += 1
            invalid.append({
                "id": record_id,
                "review_text": review_text,
                "predicted_value": str(item)[:100],
                "reason": "span_not_dict",
            })
            continue

        start_tok = item.get("start_token")
        end_tok   = item.get("end_token")

        # Check missing
        if start_tok is None or end_tok is None:
            stats["missing_field"] += 1
            invalid.append({
                "id": record_id,
                "review_text": review_text,
                "predicted_value": f"start={start_tok}, end={end_tok}",
                "reason": "missing_start_or_end_token",
            })
            continue

        # Check integer type
        if not isinstance(start_tok, int) or not isinstance(end_tok, int):
            stats["not_integer"] += 1
            invalid.append({
                "id": record_id,
                "review_text": review_text,
                "predicted_value": f"start={start_tok}({type(start_tok).__name__}), end={end_tok}({type(end_tok).__name__})",
                "reason": "token_not_integer",
            })
            continue

        # Check range
        if not (0 <= start_tok <= end_tok < n_tokens):
            stats["out_of_range"] += 1
            invalid.append({
                "id": record_id,
                "review_text": review_text,
                "predicted_value": f"start_token={start_tok}, end_token={end_tok}, n_tokens={n_tokens}",
                "reason": "token_out_of_range",
            })
            continue

        # Reconstruct from original
        start_tok_info = token_offsets[start_tok]
        end_tok_info   = token_offsets[end_tok]
        char_start = start_tok_info["start"]
        char_end   = end_tok_info["end"]
        span_text  = review_text[char_start:char_end]

        accepted.append({
            "start":         char_start,
            "end":           char_end,
            "text":          span_text,
            "start_token":   start_tok,
            "end_token":     end_tok,
            "span_type":     "complaint",
            "offset_source": "token_index_range",
        })

    return accepted, invalid, stats


# ─────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────

def deduplicate(spans: List[Dict]) -> List[Dict]:
    seen, unique = set(), []
    for s in spans:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    return unique


def resolve_nested(spans: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Remove spans fully contained within other spans. Returns (kept, removed)."""
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
# Pipeline
# ─────────────────────────────────────────────────────────

def load_pilot_candidates() -> List[Dict]:
    path = PILOT_IN / "pilot_candidates.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} pilot candidates")
    return records


def run_annotation(records: List[Dict]) -> Dict[str, Dict]:
    """Run annotation. Returns dict mapping id -> {raw_response, error}."""
    print(f"\n  Model: {MODEL}")
    print(f"  Records: {len(records)}")

    outputs = {}
    for i, rec in enumerate(records):
        if i == 0 or (i + 1) % 10 == 0 or i == len(records) - 1:
            print(f"  Annotating {i + 1}/{len(records)}...")

        tokens = tokenize(rec["text"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(rec["id"], rec["text"], tokens)},
        ]

        content, error = call_api(messages)
        outputs[rec["id"]] = {
            "raw_response": content if content else "",
            "error": error,
            "n_tokens": len(tokens),
        }

        # Interim save
        if (i + 1) % 25 == 0:
            with open(DATA_OUT / "pilot_v3_raw_outputs.jsonl", "w", encoding="utf-8") as f:
                for rid, out in outputs.items():
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"    [Saved interim: {len(outputs)}]")

        time.sleep(API_DELAY)

    return outputs


def run_validation(
    records: List[Dict],
    outputs: Dict[str, Dict],
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Validate all records."""
    validated = []
    all_invalid = []
    agg = Counter()

    for rec in records:
        rid = rec["id"]
        text = rec["text"]
        tokens = tokenize(text)
        out = outputs.get(rid, {})
        raw = out.get("raw_response", "")
        error = out.get("error")

        if not raw and error:
            agg["missing_output"] += 1
            all_invalid.append({
                "id": rid,
                "review_text": text,
                "predicted_value": error,
                "reason": "missing_output",
            })
            validated.append({"id": rid, "split": rec["split"], "text": text, "n_tokens": len(tokens), "spans": []})
            continue

        accepted, invalid, stats = validate_record(rid, text, tokens, raw)
        for k, v in stats.items():
            agg[k] += v
        all_invalid.extend(invalid)

        accepted = deduplicate(accepted)
        accepted, removed = resolve_nested(accepted)
        for r in removed:
            all_invalid.append({
                "id": rid,
                "review_text": text,
                "predicted_value": f"start_token={r.get('start_token')}, end_token={r.get('end_token')}",
                "reason": r["reason"],
            })

        validated.append({
            "id": rid,
            "split": rec["split"],
            "text": text,
            "n_tokens": len(tokens),
            "spans": accepted,
        })

    return validated, all_invalid, dict(agg)


def save_outputs(
    validated: List[Dict],
    invalid: List[Dict],
    stats: Dict,
) -> Dict:
    """Save all output files and return manifest."""

    # Validated
    with open(DATA_OUT / "pilot_v3_validated_outputs.jsonl", "w", encoding="utf-8") as f:
        for rec in validated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Raw
    with open(DATA_OUT / "pilot_v3_raw_outputs.jsonl", "w", encoding="utf-8") as f:
        pass  # saved incrementally

    # Invalid CSV
    if invalid:
        with open(DATA_OUT / "pilot_v3_invalid_outputs.csv", "w", newline="", encoding="utf-8") as f:
            fn = ["id", "reason", "predicted_value", "review_text"]
            w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
            w.writeheader()
            w.writerows(invalid)

    # Manifest
    total_spans    = sum(len(r["spans"]) for r in validated)
    records_with   = sum(1 for r in validated if r["spans"])
    records_without= sum(1 for r in validated if not r["spans"])
    avg            = total_spans / len(validated) if validated else 0

    raw_total = (
        stats.get("missing_output", 0)
        + sum(len(r["spans"]) for r in validated)
        + sum(1 for e in invalid if e.get("reason") not in ("nested_span_removed",))
    )

    manifest = {
        "pilot_records":                     len(validated),
        "records_with_spans":              records_with,
        "records_without_spans":          records_without,
        "total_raw_token_spans":          raw_total,
        "total_valid_spans":            total_spans,
        "average_valid_spans_per_record": round(avg, 3),
        "invalid_json_outputs":          stats.get("json_invalid", 0),
        "invalid_token_range_count":     (stats.get("out_of_range", 0) + stats.get("not_integer", 0)
                                         + stats.get("start_gt_end", 0) + stats.get("missing_field", 0)),
        "out_of_range_count":            stats.get("out_of_range", 0),
        "token_not_integer_count":       stats.get("not_integer", 0),
        "missing_start_or_end_count":    stats.get("missing_field", 0),
        "complaint_spans_not_list_count": stats.get("not_list", 0),
        "nested_span_removed_count":     sum(1 for e in invalid if e.get("reason") == "nested_span_removed"),
        "missing_output_records":         stats.get("missing_output", 0),
        "split_distribution":           dict(Counter(r["split"] for r in validated)),
        "model":                        MODEL,
    }

    with open(DATA_OUT / "pilot_v3_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def print_examples(validated: List[Dict], n: int = 10):
    """Print and save 10 examples."""
    shown = 0
    lines = []

    for rec in validated:
        if shown >= n:
            break
        spans = rec["spans"]
        if not spans:
            continue
        shown += 1

        lines.append(f"[{shown}] ID: {rec['id']}  ({rec['split']})")
        lines.append(f"    Review ({rec['n_tokens']} tokens): {repr(rec['text'][:80])}")
        lines.append(f"    Accepted spans ({len(spans)}):")
        for i, span in enumerate(spans[:5]):
            lines.append(f"      [{i+1}] tokens[{span['start_token']}:{span['end_token']}] "
                        f"chars[{span['start']},{span['end']}] {repr(span['text'][:60])}")
        lines.append("")

    text = "\n".join(lines)

    # Save to file
    examples_path = DATA_OUT / "pilot_v3_examples.txt"
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Step 3A.2: Token-Index Annotation Examples\n")
        f.write("=" * 70 + "\n\n")
        f.write(text)
        f.write(f"\nTotal shown: {shown}\n")

    print(f"\n  Saved examples to {examples_path.name}")

    # Print to stdout
    print(f"\n{'='*70}")
    print(f"10 EXAMPLES FOR MANUAL INSPECTION")
    print(f"{'='*70}")
    print(text)


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Step 3A.2: Token-Index Annotation — ViOCD Complaint Spans")
    print("=" * 70)
    print(f"\n  Input:    {PILOT_IN / 'pilot_candidates.jsonl'}")
    print(f"  Output:   {DATA_OUT}")
    print(f"  Model:   {MODEL}\n")

    # 1. Load
    print("-" * 70)
    print("1. LOADING PILOT CANDIDATES")
    print("-" * 70)
    records = load_pilot_candidates()

    # 2. Annotate
    print("\n" + "-" * 70)
    print("2. RUNNING TOKEN-INDEX ANNOTATION")
    print("-" * 70)
    outputs = run_annotation(records)

    # Save raw
    with open(DATA_OUT / "pilot_v3_raw_outputs.jsonl", "w", encoding="utf-8") as f:
        for rid, out in outputs.items():
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\n  Saved pilot_v3_raw_outputs.jsonl: {len(outputs)} records")

    # 3. Validate
    print("\n" + "-" * 70)
    print("3. VALIDATING AND POST-PROCESSING")
    print("-" * 70)
    validated, invalid, stats = run_validation(records, outputs)

    print(f"  Validated records:         {len(validated)}")
    print(f"  Records with spans:       {sum(1 for r in validated if r['spans'])}")
    print(f"  Total valid spans:       {sum(len(r['spans']) for r in validated)}")
    print(f"  Invalid entries:         {len(invalid)}")
    print(f"    json_invalid:         {stats.get('json_invalid', 0)}")
    print(f"    out_of_range:        {stats.get('out_of_range', 0)}")
    print(f"    not_integer:         {stats.get('not_integer', 0)}")
    print(f"    missing_field:       {stats.get('missing_field', 0)}")
    print(f"    not_list:           {stats.get('not_list', 0)}")
    print(f"    nested_removed:     {sum(1 for e in invalid if e.get('reason') == 'nested_span_removed')}")
    print(f"    missing_output:      {stats.get('missing_output', 0)}")

    # 4. Save
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
    for k, v in manifest.items():
        print(f"  {k}: {v}")

    # Success criteria
    inv_count = (manifest["invalid_token_range_count"]
                 + manifest["invalid_json_outputs"])
    raw = manifest["total_raw_token_spans"]
    ratio = (inv_count / raw * 100) if raw > 0 else 0

    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*70}")
    ok_json  = "  OK" if manifest["invalid_json_outputs"] == 0 else "  FAIL"
    ok_miss = "  OK" if manifest["missing_output_records"] == 0 else "  FAIL"
    ok_inv  = f"  OK ({ratio:.1f}%)" if ratio <= 5 else f"  WARN ({ratio:.1f}%)"
    print(f"  invalid_json_outputs = 0:        {ok_json} ({manifest['invalid_json_outputs']})")
    print(f"  missing_output_records = 0:      {ok_miss} ({manifest['missing_output_records']})")
    print(f"  invalid_token_range / raw <= 5%: {ok_inv} ({inv_count}/{raw})")

    print(f"\n  [Output files in {DATA_OUT}]")
    for f in sorted(DATA_OUT.iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".csv", ".json", ".txt"):
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  ✅ Step 3A.2 complete.")


if __name__ == "__main__":
    main()
