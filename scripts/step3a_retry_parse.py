#!/usr/bin/env python3
"""Retry script for parse-error records with robust JSON extraction."""
import json, re, warnings, requests, time

warnings.filterwarnings("ignore")

API_KEY = "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"
URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL   = "mistralai/mistral-small-4-119b-2603"

SYS = (
    "Bạn là người gán nhãn đoạn phàn nàn trong đánh giá tiếng Việt.\n"
    "Trích xuất các đoạn phàn nàn.\n"
    "Trả về CHỈ JSON array, không markdown:\n"
    '[{"id": "...", "complaint_spans": [{"text": "..."}]}]'
)


def parse_json_robust(content):
    """Extract JSON by bracket matching, handling newlines inside strings."""
    content = content.strip()
    content = re.sub(r"```json?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```\s*$", "", content)

    start = content.find("[")
    if start == -1:
        start = content.find("{")
    if start == -1:
        raise ValueError("No JSON found")

    bracket = content[start]
    close = "]" if bracket == "[" else "}"
    depth = 0
    end = start
    for i, c in enumerate(content[start:], start):
        if c == close:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        elif c == bracket:
            depth += 1

    return json.loads(content[start:end])


def call_api(record):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user",   "content": f'ID: {record["id"]}\nText: {record["text"]}\n'},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    resp = requests.post(
        URL, json=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        timeout=60,
    )
    if resp.status_code != 200:
        return {
            "id": record["id"],
            "raw_response": "",
            "error": resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}"),
        }

    content = resp.json()["choices"][0]["message"]["content"]

    try:
        parsed = parse_json_robust(content)
        if isinstance(parsed, dict):
            return {"id": record["id"], "raw_response": json.dumps(parsed, ensure_ascii=False), "error": None}
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and item.get("id") == record["id"]:
                    return {"id": record["id"], "raw_response": json.dumps(item, ensure_ascii=False), "error": None}
            if parsed == []:
                return {"id": record["id"], "raw_response": "[]", "error": None}
            return {"id": record["id"], "raw_response": json.dumps(parsed, ensure_ascii=False)[:500], "error": "id_not_found"}
        else:
            return {"id": record["id"], "raw_response": content[:500], "error": "unexpected_structure"}
    except Exception as e:
        return {"id": record["id"], "raw_response": content[:1000], "error": str(e)}


def main():
    DATA_OUT = "data/complaint_span_annotations/pilot"

    # Load current results
    with open(f"{DATA_OUT}/pilot_raw_outputs.jsonl") as f:
        raw_by_id = {json.loads(l)["id"]: json.loads(l) for l in f}

    # Find remaining parse-error records
    parse_error_ids = [
        rid for rid, r in raw_by_id.items()
        if r.get("error") == "json_parse_error"
    ]
    print(f"Retry {len(parse_error_ids)} parse-error records")

    # Load candidates
    with open(f"{DATA_OUT}/pilot_candidates.jsonl") as f:
        cand_by_id = {json.loads(l)["id"]: json.loads(l) for l in f}

    success = 0
    for i, rid in enumerate(parse_error_ids):
        rec = cand_by_id.get(rid)
        if not rec:
            print(f"  [{i+1}] {rid}: candidate not found")
            continue
        print(f"  [{i+1}/{len(parse_error_ids)}] Retrying {rid}...", end=" ", flush=True)
        result = call_api(rec)
        raw_by_id[rid] = result
        if not result.get("error") or result["error"] == "id_not_found":
            success += 1
            spans = []
            try:
                parsed = json.loads(result["raw_response"])
                if isinstance(parsed, dict):
                    spans = parsed.get("complaint_spans", [])
                elif isinstance(parsed, list):
                    spans = parsed[0].get("complaint_spans", []) if parsed else []
            except:
                pass
            print(f"OK ({len(spans)} spans)")
        else:
            print(f"FAIL: {result['error'][:80]}")
        time.sleep(0.3)

    # Save
    with open(f"{DATA_OUT}/pilot_raw_outputs.jsonl", "w") as f:
        for out in raw_by_id.values():
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    remaining = sum(1 for r in raw_by_id.values() if r.get("error") == "json_parse_error")
    print(f"\nRetry complete: {success}/{len(parse_error_ids)} fixed")
    print(f"Remaining parse errors: {remaining}")


if __name__ == "__main__":
    main()
