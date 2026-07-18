#!/usr/bin/env python3
"""
Step 3A.2 v2: Fix token-index parser to handle [start, end] array format.
Re-processes raw outputs with corrected validation logic.
"""

import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

DATA_OUT = Path("data/complaint_span_annotations/pilot_v3_token_index")


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


def extract_json(content: str) -> Optional[dict]:
    content = content.strip()
    try:
        return json.loads(content)
    except:
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
        except:
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
        except:
            continue
    return None


def safe_to_int(v):
    """Try to convert value to int, return None if impossible."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_token_span(item) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Parse a token span that could be in either format:
      {"start_token": N, "end_token": N}  or  [N, N]
    Returns (start_token, end_token, error_reason).
    """
    if isinstance(item, dict):
        start_tok = item.get("start_token")
        end_tok   = item.get("end_token")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        start_tok = safe_to_int(item[0])
        end_tok   = safe_to_int(item[1])
    else:
        return None, None, "invalid_span_format"

    if start_tok is None or end_tok is None:
        return None, None, "token_not_integer"
    if not isinstance(start_tok, int) or not isinstance(end_tok, int):
        return None, None, "token_not_integer"
    if start_tok < 0 or end_tok < 0:
        return None, None, "negative_index"

    return start_tok, end_tok, None


def validate_record(record_id, review_text, tokens, raw_response):
    """Validate token-index output, handle both [s,e] and {start_token,end_token} formats."""
    stats = Counter()
    invalid = []
    accepted = []
    n_tokens = len(tokens)
    token_offsets = {tok["idx"]: tok for tok in tokens}

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
        start_tok, end_tok, err = parse_token_span(item)

        if err:
            stats[err] += 1
            invalid.append({
                "id": record_id,
                "review_text": review_text,
                "predicted_value": str(item)[:100],
                "reason": err,
            })
            continue

        if start_tok > end_tok:
            stats["start_gt_end"] += 1
            invalid.append({
                "id": record_id,
                "review_text": review_text,
                "predicted_value": f"[{start_tok}, {end_tok}]",
                "reason": "start_token_gt_end_token",
            })
            continue

        if not (0 <= start_tok <= end_tok < n_tokens):
            stats["out_of_range"] += 1
            invalid.append({
                "id": record_id,
                "review_text": review_text,
                "predicted_value": f"start={start_tok}, end={end_tok}, n={n_tokens}",
                "reason": "token_out_of_range",
            })
            continue

        start_info = token_offsets[start_tok]
        end_info   = token_offsets[end_tok]
        char_start = start_info["start"]
        char_end   = end_info["end"]
        span_text  = review_text[char_start:char_end]

        accepted.append({
            "start": char_start, "end": char_end,
            "text": span_text,
            "start_token": start_tok, "end_token": end_tok,
            "span_type": "complaint", "offset_source": "token_index_range",
        })

    return accepted, invalid, stats


def deduplicate(spans):
    seen, unique = set(), []
    for s in spans:
        if s["text"] not in seen:
            seen.add(s["text"]); unique.append(s)
    return unique


def resolve_nested(spans):
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


def main():
    print("=" * 70)
    print("Step 3A.2 v2: Re-processing with fixed token-span parser")
    print("=" * 70)

    import time, requests

    API_KEY = "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"
    URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
    MODEL   = "mistralai/mistral-small-4-119b-2603"

    SYSTEM = """You are annotating Vietnamese customer reviews for complaint span extraction.

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

    # Load candidates
    with open("data/complaint_span_annotations/pilot/pilot_candidates.jsonl") as f:
        candidates = [json.loads(l) for l in f]
    print(f"  Loaded {len(candidates)} candidates")

    # Re-annotate all records
    print(f"  Re-annotating {len(candidates)} records...")
    raw_by_id = {}
    for i, rec in enumerate(candidates):
        if i == 0 or (i + 1) % 10 == 0 or i == len(candidates) - 1:
            print(f"    {i + 1}/{len(candidates)}...")
        tokens = tokenize(rec["text"])
        token_lines = "\n".join(f"[{t['idx']}] {t['text']}" for t in tokens)
        user_prompt = (
            f"Review ID:\n{rec['id']}\n\n"
            f"Original review:\n<review>\n{rec['text']}\n</review>\n\n"
            f"Indexed tokens:\n{token_lines}\n\n"
            f'Return exactly:\n{{\n  "id": "{rec["id"]}",\n  "complaint_token_spans": [\n    {{\n      "start_token": 0,\n      "end_token": 2\n    }}\n  ]\n}}'
        )
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        try:
            resp = requests.post(URL, json=payload,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                raw_by_id[rec["id"]] = {"raw_response": content, "error": None, "n_tokens": len(tokens)}
            else:
                raw_by_id[rec["id"]] = {"raw_response": "", "error": resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}"), "n_tokens": len(tokens)}
        except Exception as e:
            raw_by_id[rec["id"]] = {"raw_response": "", "error": str(e), "n_tokens": len(tokens)}
        time.sleep(0.3)

    # Save raw
    with open(DATA_OUT / "pilot_v3_raw_outputs.jsonl", "w") as f:
        for out in raw_by_id.values():
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Saved raw outputs: {len(raw_by_id)}")

    # Validate
    print("\n  Validating...")
    validated = []
    all_invalid = []
    agg = Counter()

    for rec in candidates:
        tokens = tokenize(rec["text"])
        out = raw_by_id.get(rec["id"], {})
        raw = out.get("raw_response", "")
        error = out.get("error")

        if not raw and error:
            agg["missing_output"] += 1
            all_invalid.append({"id": rec["id"], "review_text": rec["text"],
                               "predicted_value": error, "reason": "missing_output"})
            validated.append({"id": rec["id"], "split": rec["split"], "text": rec["text"],
                             "n_tokens": len(tokens), "spans": []})
            continue

        accepted, invalid, stats = validate_record(rec["id"], rec["text"], tokens, raw)
        for k, v in stats.items():
            agg[k] += v
        all_invalid.extend(invalid)

        accepted = deduplicate(accepted)
        accepted, removed = resolve_nested(accepted)
        for r in removed:
            all_invalid.append({"id": rec["id"], "review_text": rec["text"],
                               "predicted_value": f"[{r.get('start_token')}, {r.get('end_token')}]",
                               "reason": r["reason"]})

        validated.append({"id": rec["id"], "split": rec["split"], "text": rec["text"],
                        "n_tokens": len(tokens), "spans": accepted})

    # Save validated
    with open(DATA_OUT / "pilot_v3_validated_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in validated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save invalid CSV
    if all_invalid:
        with open(DATA_OUT / "pilot_v3_invalid_outputs.csv", "w", newline="", encoding="utf-8") as f:
            fn = ["id", "reason", "predicted_value", "review_text"]
            w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
            w.writeheader(); w.writerows(all_invalid)

    # Manifest
    total_spans    = sum(len(r["spans"]) for r in validated)
    records_with   = sum(1 for r in validated if r["spans"])
    records_without= sum(1 for r in validated if not r["spans"])
    avg            = total_spans / len(validated) if validated else 0

    manifest = {
        "pilot_records":                     len(validated),
        "records_with_spans":              records_with,
        "records_without_spans":          records_without,
        "total_raw_token_spans":        sum(len(r["spans"]) for r in validated) + sum(1 for e in all_invalid if e.get("reason") not in ("nested_span_removed",)),
        "total_valid_spans":            total_spans,
        "average_valid_spans_per_record":   round(avg, 3),
        "invalid_json_outputs":          agg["json_invalid"],
        "invalid_token_range_count":     agg["out_of_range"] + agg["not_integer"] + agg["start_gt_end"] + agg["negative_index"] + agg["invalid_span_format"] + agg["missing_output"],
        "out_of_range_count":           agg["out_of_range"],
        "token_not_integer_count":       agg["not_integer"],
        "invalid_span_format_count":     agg.get("invalid_span_format", 0),
        "nested_span_removed_count":    sum(1 for e in all_invalid if e.get("reason") == "nested_span_removed"),
        "missing_output_records":         agg["missing_output"],
        "split_distribution":           dict(Counter(r["split"] for r in validated)),
        "model":                        MODEL,
    }

    with open(DATA_OUT / "pilot_v3_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Examples
    with open(DATA_OUT / "pilot_v3_examples.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Step 3A.2: Token-Index Annotation Examples (v2)\n")
        f.write("=" * 70 + "\n\n")
        shown = 0
        for rec in validated:
            if shown >= 10 or not rec["spans"]:
                continue
            shown += 1
            f.write(f"[{shown}] ID: {rec['id']}  ({rec['split']})\n")
            f.write(f"    Review ({rec['n_tokens']} tokens): {repr(rec['text'][:80])}\n")
            f.write(f"    Spans ({len(rec['spans'])}):\n")
            for i, s in enumerate(rec["spans"][:5]):
                f.write(f"      [{i+1}] tokens[{s['start_token']}:{s['end_token']}] "
                       f"chars[{s['start']},{s['end']}] {repr(s['text'][:60])}\n")
            f.write("\n")
        f.write(f"Total shown: {shown}\n")

    # Print
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Validated records:         {len(validated)}")
    print(f"  Records with spans:       {records_with}")
    print(f"  Total valid spans:       {total_spans}")
    print(f"  Avg spans/record:        {avg:.2f}")
    print(f"  json_invalid:            {agg['json_invalid']}")
    print(f"  out_of_range:           {agg['out_of_range']}")
    print(f"  not_integer:            {agg['not_integer']}")
    print(f"  invalid_span_format:    {agg.get('invalid_span_format', 0)}")
    print(f"  nested_removed:        {sum(1 for e in all_invalid if e.get('reason') == 'nested_span_removed')}")
    print(f"  missing_output:         {agg['missing_output']}")

    print(f"\n{'='*70}")
    print("MANIFEST")
    print(f"{'='*70}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    ratio = (manifest["invalid_token_range_count"] / manifest["total_raw_token_spans"] * 100) if manifest["total_raw_token_spans"] > 0 else 0
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*70}")
    ok_json  = "  OK" if manifest["invalid_json_outputs"] == 0 else "  FAIL"
    ok_miss  = "  OK" if manifest["missing_output_records"] == 0 else "  FAIL"
    ok_ratio = f"  OK ({ratio:.1f}%)" if ratio <= 5 else f"  WARN ({ratio:.1f}%)"
    print(f"  invalid_json_outputs = 0:       {ok_json} ({manifest['invalid_json_outputs']})")
    print(f"  missing_output_records = 0:     {ok_miss} ({manifest['missing_output_records']})")
    print(f"  invalid / raw <= 5%:          {ok_ratio} ({manifest['invalid_token_range_count']}/{manifest['total_raw_token_spans']})")

    print(f"\n  [Output files in {DATA_OUT}]")
    for f in sorted(DATA_OUT.iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".csv", ".json", ".txt"):
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  ✅ Step 3A.2 v2 complete.")


if __name__ == "__main__":
    main()
