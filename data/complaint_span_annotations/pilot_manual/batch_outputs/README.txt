Batch LLM Annotation Output Directory
======================================
Place your LLM JSON outputs here after running each batch prompt.

Expected filenames:
  batch_001_output.json
  batch_002_output.json
  batch_003_output.json
  batch_004_output.json

Each file must contain a JSON array with this structure:
[
  {
    "id": "record_id",
    "complaint_spans": [
      {"text": "complaint expression 1"},
      {"text": "complaint expression 2"}
    ]
  },
  ...
]

Steps:
1. Open batch_001_prompt.txt in a text editor.
2. Copy the entire contents.
3. Paste into your LLM web UI (e.g., ChatGPT, Claude).
4. Wait for the JSON array response.
5. Copy the JSON response.
6. Save as batch_001_output.json in this folder.
7. Repeat for batches 002, 003, and 004.

Then run:
  python3 validate_manual_outputs.py

The validator will:
- Load all batch_*_output.json files
- Validate each span text appears in the original review
- Compute character offsets via exact string matching
- Remove duplicates, empty spans, and invalid entries
- Save results to pilot_manual_validated.jsonl and pilot_manual_manifest.json
