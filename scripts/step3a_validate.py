#!/usr/bin/env python3
"""Re-validate all pilot outputs and generate final manifest."""
import csv, json, warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_OUT = Path("data/complaint_span_annotations/pilot")

def find_span_offset(text, span_text):
    if not span_text or not isinstance(span_text, str): return None
    span_text = span_text.strip()
    if not span_text: return None
    idx = text.find(span_text)
    return (idx, idx + len(span_text)) if idx >= 0 else None

def is_invalid_span_text(text):
    if not text or not isinstance(text, str): return True
    stripped = text.strip()
    if not stripped: return True
    import string
    punct = set(string.punctuation)
    return all(c in punct | {" ", "\t", "\n", "\r", "\u00a0"} for c in stripped)

def remove_dupes(spans):
    seen, unique = set(), []
    for s in spans:
        if s["text"] not in seen:
            seen.add(s["text"]); unique.append(s)
    return unique

def resolve_nested(spans):
    if len(spans) <= 1: return spans
    sorted_spans = sorted(spans, key=lambda s: (s["end"] - s["start"], s["start"]))
    resolved = []
    for span in sorted_spans:
        if any(e["start"] <= span["start"] and span["end"] <= e["end"] for e in resolved):
            continue
        resolved = [e for e in resolved if not (span["start"] <= e["start"] and e["end"] <= span["end"])]
        resolved.append(span)
    return resolved

def parse_and_validate(record, raw):
    rid, text = record["id"], record["text"]
    validated, invalid = [], []

    parsed = None
    if raw and raw != "[]":
        try: parsed = json.loads(raw)
        except: parsed = None

    if parsed is None:
        if raw != "[]" and raw:
            invalid.append({"record_id": rid, "split": record["split"], "reason": "json_invalid", "text": text, "raw": raw[:300]})
        return [], invalid

    spans = parsed.get("complaint_spans", []) if isinstance(parsed, dict) else []
    if not isinstance(spans, list):
        invalid.append({"record_id": rid, "split": record["split"], "reason": "spans_not_list", "text": text})
        return [], invalid

    for idx, span in enumerate(spans):
        if not isinstance(span, dict):
            invalid.append({"record_id": rid, "split": record["split"], "reason": "span_not_dict", "span_index": idx, "text": str(span)[:100], "review_text": text})
            continue
        span_text = span.get("text", "")
        if is_invalid_span_text(span_text):
            invalid.append({"record_id": rid, "split": record["split"], "reason": "invalid_span_text", "span_index": idx, "text": span_text[:100], "review_text": text})
            continue
        offset = find_span_offset(text, span_text)
        if offset is None:
            invalid.append({"record_id": rid, "split": record["split"], "reason": "text_not_in_review", "span_index": idx, "text": span_text[:200], "review_text": text})
            continue
        validated.append({"text": span_text, "start": offset[0], "end": offset[1]})

    validated = remove_dupes(validated)
    validated = resolve_nested(validated)
    return validated, invalid

# Load
with open(DATA_OUT / "pilot_candidates.jsonl") as f:
    candidates = {json.loads(l)["id"]: json.loads(l) for l in f}

with open(DATA_OUT / "pilot_raw_outputs.jsonl") as f:
    raw_by_id = {json.loads(l)["id"]: json.loads(l) for l in f}

# Validate all
validated_records = []
all_invalid = []
for rid, rec in candidates.items():
    raw = raw_by_id.get(rid, {}).get("raw_response", "")
    spans, invalid = parse_and_validate(rec, raw)
    for e in invalid:
        all_invalid.append(e)
    # Records with no spans are still valid (empty complaint)
    validated_records.append({"id": rid, "split": rec["split"], "text": rec["text"], "spans": spans})

# Stats
total_spans = sum(len(r["spans"]) for r in validated_records)
records_with = sum(1 for r in validated_records if r["spans"])
records_without = sum(1 for r in validated_records if not r["spans"])
avg = total_spans / len(validated_records) if validated_records else 0
split_dist = dict(Counter(r["split"] for r in validated_records))

print(f"Validated: {len(validated_records)}")
print(f"With spans: {records_with}")
print(f"Without spans: {records_without}")
print(f"Total spans: {total_spans}")
print(f"Avg spans: {avg:.2f}")
print(f"Invalid: {len(all_invalid)}")

# Save validated
with open(DATA_OUT / "pilot_validated_outputs.jsonl", "w") as f:
    for r in validated_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Save invalid
if all_invalid:
    with open(DATA_OUT / "pilot_invalid_outputs.csv", "w", newline="") as f:
        fn = ["record_id", "split", "reason", "span_index", "text"]
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader(); w.writerows(all_invalid)

# Save manifest
manifest = {
    "pilot_records": len(validated_records),
    "records_with_spans": records_with,
    "records_without_spans": records_without,
    "total_spans": total_spans,
    "average_spans_per_record": round(avg, 3),
    "invalid_output_count": len(all_invalid),
    "invalid_span_text_count": sum(1 for e in all_invalid if e["reason"] == "invalid_span_text"),
    "text_not_found_count": sum(1 for e in all_invalid if e["reason"] == "text_not_in_review"),
    "json_parse_error_count": sum(1 for e in all_invalid if e["reason"] in ("json_invalid", "spans_not_list")),
    "split_distribution": split_dist,
    "model": "mistralai/mistral-small-4-119b-2603",
}
with open(DATA_OUT / "pilot_manifest.json", "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("\nManifest:")
print(json.dumps(manifest, indent=2, ensure_ascii=False))

# Print 10 examples
print("\n" + "="*60)
print("10 EXAMPLES FOR MANUAL INSPECTION")
print("="*60)
shown = 0
for rec in validated_records:
    if shown >= 10 or not rec["spans"]: continue
    shown += 1
    print(f"\n[{shown}] ID: {rec['id']} ({rec['split']})")
    print(f"    Text: {repr(rec['text'][:80])}")
    for i, s in enumerate(rec["spans"][:5]):
        print(f"    [{i+1}] [{s['start']},{s['end']}] {repr(s['text'][:60])}")

print("\n✅ Validation complete")
