#!/usr/bin/env python3
"""
Step 3A.1 v2: Add safe whitespace normalization repair to the pilot results.
Re-processes pilot_v2_validated_outputs.jsonl with improved post-processing.

Safe repair order:
A. Exact match
B. Trim wrapping chars
C. Trim terminal punctuation
D. Normalize internal whitespace (collapse double spaces) — SAFE repair
E. Otherwise reject
"""

import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_OUT = Path("data/complaint_span_annotations/pilot_v2")


def normalize_internal_whitespace(text: str) -> str:
    """Collapse multiple spaces into single spaces (safe normalization)."""
    return re.sub(r"  +", " ", text)


def safe_repair(span_text: str, review_text: str) -> tuple:
    """
    Try safe repairs on span_text to find exact match in review_text.
    Returns (repaired_text, repair_type) or (None, None).
    """
    # A. Exact match
    if span_text in review_text:
        return span_text, None

    # B. Trim wrapping chars
    trimmed = span_text.strip(" \t\n\r\x0b\x0c\u00a0\"'`")
    if trimmed in review_text:
        return trimmed, "trim_wrapping_chars"

    # C. Trim terminal punctuation
    for _ in range(3):
        trimmed2 = trimmed.rstrip(".,;:!?…\u2026")
        if trimmed2 in review_text:
            return trimmed2, "trim_extra_terminal_punctuation"
        if trimmed2 == trimmed:
            break
        trimmed = trimmed2

    # D. Normalize internal whitespace (collapse double spaces)
    # This handles cases where the review has double spaces and the LLM collapsed them
    norm_span = normalize_internal_whitespace(span_text)
    if norm_span != span_text and norm_span in review_text:
        return norm_span, "normalize_internal_whitespace"

    return None, None


def main():
    print("=" * 70)
    print("Step 3A.1: Re-processing with whitespace normalization repair")
    print("=" * 70)

    # Load raw outputs (reconstruct from validated + invalid)
    # We need to re-parse from scratch using the raw outputs
    # The raw outputs file was empty, so we need to re-annotate or use validated + invalid

    # Since raw_outputs.jsonl is empty, we need to re-annotate the 100 records
    # OR we can reconstruct from the validated + invalid entries
    # Best approach: re-annotate just the 100 records with improved prompt

    print("\n  Re-annotating 100 records with improved post-processing...")

    import time, requests

    API_KEY = "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"
    URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    MODEL = "mistralai/mistral-small-4-119b-2603"

    SYSTEM = """You are annotating Vietnamese customer reviews for complaint span extraction.

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

    USER_TEMPLATE = """Review ID:
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

    def extract_json(content):
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

    # Load candidates
    with open(DATA_OUT.parent / "pilot" / "pilot_candidates.jsonl") as f:
        candidates = [json.loads(l) for l in f]
    print(f"  Loaded {len(candidates)} candidates")

    # Check if we already have raw outputs
    raw_path = DATA_OUT / "pilot_v2_raw_outputs.jsonl"
    raw_exists = raw_path.exists() and raw_path.stat().st_size > 0

    if raw_exists:
        with open(raw_path) as f:
            raw_by_id = {json.loads(l)["id"]: json.loads(l) for l in f}
        print(f"  Loaded {len(raw_by_id)} raw outputs from file")
    else:
        # Re-annotate
        print(f"  Re-annotating all {len(candidates)} records...")
        raw_by_id = {}
        for i, rec in enumerate(candidates):
            if i == 0 or (i + 1) % 10 == 0 or i == len(candidates) - 1:
                print(f"    {i + 1}/{len(candidates)}...")
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": USER_TEMPLATE.format(id=rec["id"], text=rec["text"])},
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
                    raw_by_id[rec["id"]] = {"raw_response": content, "error": None}
                else:
                    raw_by_id[rec["id"]] = {"raw_response": "", "error": resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")}
            except Exception as e:
                raw_by_id[rec["id"]] = {"raw_response": "", "error": str(e)}
            time.sleep(0.3)

        # Save raw
        with open(raw_path, "w", encoding="utf-8") as f:
            for out in raw_by_id.values():
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  Saved raw outputs: {len(raw_by_id)} records")

    # ── Validate with improved post-processing ──────────────────────────
    print("\n  Validating with whitespace normalization repair...")

    validated = []
    all_invalid = []
    stats = Counter()

    for rec in candidates:
        rid = rec["id"]
        text = rec["text"]
        out = raw_by_id.get(rid, {})
        raw = out.get("raw_response", "")
        error = out.get("error")

        if not raw and error:
            stats["missing_output"] += 1
            all_invalid.append({"id": rid, "review_text": text, "predicted_span_text": "", "reason": f"missing: {error}"})
            validated.append({"id": rid, "split": rec["split"], "text": text, "spans": []})
            continue

        parsed = extract_json(raw)
        if parsed is None:
            stats["json_invalid"] += 1
            all_invalid.append({"id": rid, "review_text": text, "predicted_span_text": "", "reason": "json_invalid"})
            validated.append({"id": rid, "split": rec["split"], "text": text, "spans": []})
            continue

        complaint_spans = []
        if isinstance(parsed, dict):
            complaint_spans = parsed.get("complaint_spans", [])
        elif isinstance(parsed, list) and parsed:
            complaint_spans = parsed[0].get("complaint_spans", []) if isinstance(parsed[0], dict) else []

        if not isinstance(complaint_spans, list):
            stats["not_list"] += 1
            validated.append({"id": rid, "split": rec["split"], "text": text, "spans": []})
            continue

        if len(complaint_spans) == 0:
            stats["empty"] += 1
            validated.append({"id": rid, "split": rec["split"], "text": text, "spans": []})
            continue

        accepted = []
        for idx, span in enumerate(complaint_spans):
            if not isinstance(span, dict):
                stats["span_not_dict"] += 1
                all_invalid.append({"id": rid, "review_text": text, "predicted_span_text": str(span)[:100], "reason": "span_not_dict"})
                continue

            span_text = span.get("text", "")
            if not isinstance(span_text, str) or not span_text.strip():
                stats["empty_span"] += 1
                all_invalid.append({"id": rid, "review_text": text, "predicted_span_text": repr(span_text)[:50], "reason": "empty_span"})
                continue

            repaired_text, repair_type = safe_repair(span_text, text)

            if repaired_text is None:
                stats["text_not_found"] += 1
                all_invalid.append({"id": rid, "review_text": text, "predicted_span_text": span_text[:200], "reason": "text_not_in_review"})
                continue

            if repair_type == "normalize_internal_whitespace":
                stats["whitespace_norm"] += 1
            elif repair_type == "trim_wrapping_chars":
                stats["trim_wrapping"] += 1
            elif repair_type == "trim_extra_terminal_punctuation":
                stats["terminal_punct"] += 1

            start = text.find(repaired_text)
            accepted.append({"text": repaired_text, "start": start, "end": start + len(repaired_text), "repair_type": repair_type})

        # Deduplicate
        seen, unique = set(), []
        for s in accepted:
            if s["text"] not in seen:
                seen.add(s["text"]); unique.append(s)
        accepted = unique

        # Resolve nested
        if len(accepted) > 1:
            sorted_spans = sorted(accepted, key=lambda s: (s["end"] - s["start"], s["start"]))
            kept, removed = [], []
            for span in sorted_spans:
                if any(e["start"] <= span["start"] and span["end"] <= e["end"] for e in kept):
                    removed.append({**span, "reason": "nested_span_removed"})
                    continue
                kept = [e for e in kept if not (span["start"] <= e["start"] and e["end"] <= span["end"])]
                kept.append(span)
            accepted = kept
            for r in removed:
                all_invalid.append({"id": rid, "review_text": text, "predicted_span_text": r["text"], "reason": r["reason"]})

        validated.append({"id": rid, "split": rec["split"], "text": text, "spans": accepted})

    # ── Save outputs ───────────────────────────────────────────────────
    total_spans = sum(len(r["spans"]) for r in validated)
    records_with = sum(1 for r in validated if r["spans"])
    records_without = sum(1 for r in validated if not r["spans"])
    avg = total_spans / len(validated) if validated else 0

    # Raw spans calculation
    raw_total = (stats["empty"] + stats["whitespace_norm"] + stats["trim_wrapping"]
               + stats["terminal_punct"] + stats["text_not_found"] + stats["span_not_dict"]
               + sum(len(r["spans"]) for r in validated))

    manifest = {
        "pilot_records":                       len(validated),
        "records_with_spans":                records_with,
        "records_without_spans":            records_without,
        "total_raw_spans":                 raw_total,
        "total_valid_spans":              total_spans,
        "average_valid_spans_per_record":   round(avg, 3),
        "invalid_json_outputs":            stats["json_invalid"],
        "span_text_not_found_count":        stats["text_not_found"],
        "span_not_dict_count":            stats.get("span_not_dict", 0),
        "empty_span_count":               stats["empty_span"],
        "trim_wrapping_repairs":          stats["trim_wrapping"],
        "terminal_punctuation_repairs":   stats["terminal_punct"],
        "whitespace_normalization_repairs": stats["whitespace_norm"],
        "nested_span_removed_count":      sum(1 for e in all_invalid if e.get("reason") == "nested_span_removed"),
        "missing_output_records":         stats["missing_output"],
        "split_distribution":             dict(Counter(r["split"] for r in validated)),
        "model":                        MODEL,
    }

    # Save validated
    with open(DATA_OUT / "pilot_v2_validated_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in validated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save invalid
    if all_invalid:
        with open(DATA_OUT / "pilot_v2_invalid_outputs.csv", "w", newline="", encoding="utf-8") as f:
            fn = ["id", "reason", "predicted_span_text", "review_text"]
            w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
            w.writeheader(); w.writerows(all_invalid)

    # Save manifest
    with open(DATA_OUT / "pilot_v2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Save examples
    examples_path = DATA_OUT / "pilot_v2_examples.txt"
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Step 3A.1: Pilot Annotation Examples (v2 - with whitespace repair)\n")
        f.write("=" * 70 + "\n\n")
        shown = 0
        for rec in validated:
            if shown >= 10 or not rec["spans"]:
                continue
            shown += 1
            f.write(f"[{shown}] ID: {rec['id']}  ({rec['split']})\n")
            f.write(f"    Review: {repr(rec['text'])}\n")
            f.write(f"    Accepted spans ({len(rec['spans'])}):\n")
            for i, span in enumerate(rec["spans"][:5]):
                repair = f" [repaired: {span.get('repair_type', 'exact')}]" if span.get("repair_type") else ""
                f.write(f"      [{i+1}] [{span['start']},{span['end']}] {repr(span['text'][:60])}{repair}\n")
            f.write("\n")
        f.write(f"Total shown: {shown}\n")

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Validated records:          {len(validated)}")
    print(f"  Records with spans:         {records_with}")
    print(f"  Total valid spans:          {total_spans}")
    print(f"  Avg spans/record:           {avg:.2f}")
    print(f"\n  invalid_json_outputs:       {stats['json_invalid']}")
    print(f"  text_not_found:            {stats['text_not_found']}")
    print(f"  span_not_dict:             {stats.get('span_not_dict', 0)}")
    print(f"  empty_span:                {stats['empty_span']}")
    print(f"  trim_wrapping:             {stats['trim_wrapping']}")
    print(f"  terminal_punct:            {stats['terminal_punct']}")
    print(f"  whitespace_norm:            {stats['whitespace_norm']}")
    print(f"  missing_output:             {stats['missing_output']}")
    print(f"  nested_removed:            {sum(1 for e in all_invalid if e.get('reason') == 'nested_span_removed')}")

    print(f"\n{'='*70}")
    print("MANIFEST")
    print(f"{'='*70}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    ratio = (stats['text_not_found'] / raw_total * 100) if raw_total > 0 else 0
    print(f"\n{'='*70}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*70}")
    ok_json   = "  OK" if stats['json_invalid'] == 0 else "  FAIL"
    ok_miss   = "  OK" if stats['missing_output'] == 0 else "  FAIL"
    ok_notdict= "  OK" if stats.get('span_not_dict', 0) == 0 else "  WARN"
    ok_ratio  = f"  OK ({ratio:.1f}%)" if ratio <= 5 else f"  WARN ({ratio:.1f}%)"
    print(f"  invalid_json_outputs = 0:   {ok_json} ({stats['json_invalid']})")
    print(f"  missing_output_records = 0: {ok_miss} ({stats['missing_output']})")
    print(f"  span_not_dict_count = 0:   {ok_notdict} ({stats.get('span_not_dict', 0)})")
    print(f"  text_not_found/raw <= 5%: {ok_ratio} ({stats['text_not_found']}/{raw_total})")

    print(f"\n  [Output files in {DATA_OUT}]")
    for f in sorted(DATA_OUT.iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".csv", ".json", ".txt"):
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    print(f"\n  ✅ Step 3A.1 v2 complete.")


if __name__ == "__main__":
    main()
