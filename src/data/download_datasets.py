#!/usr/bin/env python3
"""
Download all three datasets:
  - UIT-ViSD4SA: GitHub raw files
  - CausaSent-ATE-v2: HuggingFace datasets
  - ViOCD: HuggingFace datasets
"""

import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Project root is two levels up from this file (src/data/ -> project root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists")
        return
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            block_size = 8192
            with open(dest, "wb") as f, tqdm(total=total, desc=desc, unit="B", unit_scale=True) as bar:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"  [OK] {dest.name}")
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")
        if dest.exists():
            dest.unlink()


def download_uivsd4sa():
    """Download UIT-ViSD4SA from GitHub raw files."""
    print("\n[1/3] Downloading UIT-ViSD4SA...")
    base = "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data"
    files = {
        "train.jsonl": f"{base}/train.jsonl",
        "dev.jsonl": f"{base}/dev.jsonl",
        "test.jsonl": f"{base}/test.jsonl",
    }
    out_dir = DATA_RAW / "uvisd4sa"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in files.items():
        download_file(url, out_dir / fname, f"  {fname}")


def download_viocd():
    """Download ViOCD from HuggingFace using the datasets library."""
    print("\n[2/3] Downloading ViOCD from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [WARN] datasets library not installed. Install with: pip install datasets")
        print("  [SKIP] ViOCD download")
        return

    out_dir = DATA_RAW / "viocd"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("tarudesu/ViOCD", trust_remote_code=True)
        for split, df in ds.items():
            out_path = out_dir / f"{split}.jsonl"
            if out_path.exists():
                print(f"  [SKIP] {split}.jsonl already exists")
                continue
            records = []
            for row in df:
                records.append({
                    "id": row.get("id", f"{split}_{len(records)}"),
                    "text": row.get("text", row.get("review", "")),
                    "label": row.get("label", row.get("complaint", "")),
                })
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [OK] {split}.jsonl ({len(records)} records)")
    except Exception as e:
        print(f"  [FAIL] ViOCD: {e}")


def download_causasent():
    """Download CausaSent-ATE-v2 from HuggingFace."""
    print("\n[3/3] Downloading CausaSent-ATE-v2 from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [WARN] datasets library not installed. Install with: pip install datasets")
        print("  [SKIP] CausaSent download")
        return

    out_dir = DATA_RAW / "causasent_ate_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("Tamir39/causasent-ate-v2", trust_remote_code=True)
        for split, df in ds.items():
            out_path = out_dir / f"{split}.jsonl"
            if out_path.exists():
                print(f"  [SKIP] {split}.jsonl already exists")
                continue
            records = []
            for row in df:
                rec = {
                    "id": row.get("id", ""),
                    "source": row.get("source", ""),
                    "review": row.get("review", row.get("text", "")),
                    "annotations": row.get("annotations", []),
                }
                records.append(rec)
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [OK] {split}.jsonl ({len(records)} records)")
    except Exception as e:
        print(f"  [FAIL] CausaSent-ATE-v2: {e}")


def main():
    print("=" * 60)
    print("Downloading all datasets for Vietnamese Complaint Span Extraction")
    print(f"Output directory: {DATA_RAW}")
    print("=" * 60)

    download_uivsd4sa()
    download_viocd()
    download_causasent()

    print("\n" + "=" * 60)
    print("Download complete. Check data/raw/ for files.")
    print("=" * 60)

    # Print summary
    for subdir in sorted(DATA_RAW.iterdir()):
        if subdir.is_dir():
            print(f"\n{subdir.name}/")
            for f in sorted(subdir.iterdir()):
                size = f.stat().st_size
                print(f"  {f.name}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
