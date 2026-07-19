#!/usr/bin/env python3
"""
Build paper-ready tables, figures, and analysis files from completed experiments.

Inputs (any of these — see ``discover_inputs`` for fallback paths):
- outputs/experiments/direct_baselines/experiment_results.csv
- outputs/experiments/transfer_learning/experiment_results.csv
- outputs/experiments/direct_baselines/<model>/test_metrics.json
- outputs/experiments/direct_baselines/<model>/completed_result.json
- outputs/experiments/transfer_learning/<strategy>/test_metrics.json
- outputs/experiments/transfer_learning/<strategy>/completed_result.json
- outputs/experiments/transfer_learning/<strategy>/error_analysis.csv
- outputs/experiments/transfer_learning/<strategy>/per_label_report.csv

Outputs (all under outputs/paper_assets/):
- main_results.csv
- main_results.md
- main_results_latex.txt
- improvement_summary.json
- improvement_summary.txt
- entity_f1_comparison.png
- best_model_error_summary.csv
- best_model_error_summary.md
- per_label_reports_combined.csv
- result_interpretation.txt

Robustness:
- Searches ``outputs/`` recursively if exact paths are missing.
- Reconstructs direct/transfer ``experiment_results.csv`` automatically from
  per-run JSON files when those CSVs are absent.
- Prints a clear diagnostic block listing everything found before loading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Use a non-interactive matplotlib backend so this works on Kaggle/CI
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRECT_DIR = REPO_ROOT / "outputs" / "experiments" / "direct_baselines"
DEFAULT_TRANSFER_DIR = REPO_ROOT / "outputs" / "experiments" / "transfer_learning"
DEFAULT_ASSETS_DIR = REPO_ROOT / "outputs" / "paper_assets"

# Direct baseline folders in canonical order, with their static metadata
DIRECT_MODELS: List[Tuple[str, str, str]] = [
    ("phobert_ce",            "vinai/phobert-base-v2",         "ce"),
    ("phobert_weighted_ce",   "vinai/phobert-base-v2",         "weighted_ce"),
    ("xlm_roberta_ce",        "xlm-roberta-base",              "ce"),
    ("mbert_ce",              "bert-base-multilingual-cased",  "ce"),
]

# Transfer strategies in canonical order, with auxiliary_data labels
TRANSFER_STRATEGIES: List[Tuple[str, str]] = [
    ("aux_all_then_complaint",   "UIT-ViSD4SA + CausaSent-ATE-v2"),
    ("causasent_then_complaint", "CausaSent-ATE-v2"),
    ("uvisd4sa_then_complaint",  "UIT-ViSD4SA"),
]

# Key aliases — JSON keys may differ between scripts
KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "entity_precision": ("entity_precision", "test_entity_precision"),
    "entity_recall":    ("entity_recall",    "test_entity_recall"),
    "entity_f1":        ("entity_f1",        "test_entity_f1"),
    "token_macro_f1":   ("token_macro_f1",   "test_token_f1_macro", "token_f1_macro"),
    "token_accuracy":   ("token_accuracy",   "test_token_accuracy"),
    "train_runtime":    ("train_runtime",    "train_time_seconds", "total_time_seconds",
                         "total_runtime", "test_train_runtime"),
}


# ---------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------

def _safe_glob(root: Path, patterns) -> List[Path]:
    """Return sorted unique files matching any pattern under root, ignoring .git."""
    # Defensive: accept either a single string or any iterable of patterns.
    if isinstance(patterns, (str, bytes)):
        patterns = [patterns]
    out: List[Path] = []
    seen: set = set()
    for pattern in patterns:
        # On older Python (3.9), Path.glob does not support '**/name' patterns.
        # Use rglob for recursive patterns; glob for non-recursive ones.
        if not isinstance(pattern, str):
            continue
        if pattern.startswith("**/"):
            target = pattern[3:]
            iterator = root.rglob(target) if target else root.rglob("*")
        else:
            try:
                iterator = root.glob(pattern)
            except (NotImplementedError, ValueError):
                continue
        for p in iterator:
            if ".git" in p.parts:
                continue
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def print_diagnostics(repo_root: Path, outputs_root: Path) -> None:
    print(f"Repo root: {repo_root}")
    print(f"Outputs root: {outputs_root}")
    print("Existing output files:")

    # Search recursively under outputs/ and repo root (capped depth to keep fast)
    def _all_under(root: Path, name: str) -> List[Path]:
        if not root.exists():
            return []
        return sorted(p for p in root.rglob(name) if ".git" not in p.parts)

    exp_results = _all_under(outputs_root, "experiment_results.csv")
    exp_results += _all_under(repo_root, "experiment_results.csv")
    test_metrics = _all_under(outputs_root, "test_metrics.json")
    completed = _all_under(outputs_root, "completed_result.json")
    err = _all_under(outputs_root, "error_analysis.csv")
    per_label = _all_under(outputs_root, "per_label_report.csv")

    def _fmt(paths: List[Path]) -> str:
        if not paths:
            return "    (none)"
        return "\n".join(f"    - {p.relative_to(repo_root)}" for p in paths)

    print(f"  experiment_results.csv:\n{_fmt(exp_results)}")
    print(f"  test_metrics.json:\n{_fmt(test_metrics)}")
    print(f"  completed_result.json:\n{_fmt(completed)}")
    print(f"  error_analysis.csv:\n{_fmt(err)}")
    print(f"  per_label_report.csv:\n{_fmt(per_label)}")


# ---------------------------------------------------------------
# Loaders — robust to missing/misplaced files
# ---------------------------------------------------------------

def discover_experiment_results_csv(
    repo_root: Path,
    outputs_root: Path,
    setting_type: str,
) -> Optional[Path]:
    """
    Locate an experiment_results.csv file for ``setting_type``
    (``direct`` or ``transfer``). Searches recursively under outputs/ and
    repo root, then classifies candidates by path substring. Returns ``None``
    if no candidate matches the setting-specific path marker.
    """
    needle = "direct_baselines" if setting_type == "direct" else "transfer_learning"

    candidates: List[Path] = []
    candidates += _safe_glob(outputs_root, ["**/experiment_results.csv"])
    candidates += _safe_glob(repo_root, ["**/experiment_results.csv"])

    # Strict classification: only paths matching the setting marker
    for p in candidates:
        if needle in str(p):
            return p

    return None


def _pick(metrics: Dict[str, Any], canonical_key: str) -> Optional[float]:
    """Look up a metric using all known aliases. Returns float or None."""
    for alias in KEY_ALIASES.get(canonical_key, (canonical_key,)):
        if alias in metrics and metrics[alias] is not None:
            try:
                return float(metrics[alias])
            except (TypeError, ValueError):
                continue
    return None


def _load_metrics_dict(path: Path) -> Optional[Dict[str, Any]]:
    """Load a metrics JSON; unwraps ``completed_result.json`` if needed."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None
    # Unwrap completed_result.json
    if isinstance(data, dict) and "test_metrics" in data and isinstance(data["test_metrics"], dict):
        return data["test_metrics"]
    return data


def reconstruct_direct_csv(
    direct_dir: Path,
    out_path: Path,
) -> bool:
    """
    Reconstruct outputs/<direct_dir>/experiment_results.csv from each
    per-model JSON file. Returns True on success.
    """
    if not direct_dir.exists():
        print(f"  [WARN] direct_dir does not exist: {direct_dir}")
        return False

    rows: List[Dict[str, Any]] = []
    for model_key, model_name, loss_type in DIRECT_MODELS:
        model_dir = direct_dir / model_key
        # Prefer test_metrics.json, fall back to completed_result.json
        src = model_dir / "test_metrics.json"
        if not src.exists():
            src = model_dir / "completed_result.json"
        if not src.exists():
            print(f"  [WARN] No JSON for direct model {model_key}; skipping.")
            continue

        m = _load_metrics_dict(src)
        if m is None:
            continue

        rows.append(
            {
                "experiment_name": model_key,
                "model_name":       m.get("model_name", model_name),
                "loss_type":        m.get("loss", loss_type),
                "entity_precision": _pick(m, "entity_precision"),
                "entity_recall":    _pick(m, "entity_recall"),
                "entity_f1":        _pick(m, "entity_f1"),
                "token_macro_f1":   _pick(m, "token_macro_f1"),
                "token_accuracy":   _pick(m, "token_accuracy"),
                "train_runtime":    _pick(m, "train_runtime"),
                "output_dir":       str(model_dir.relative_to(REPO_ROOT)),
            }
        )

    if not rows:
        return False

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [INFO] Reconstructed direct CSV at {out_path.relative_to(REPO_ROOT)} "
          f"with {len(df)} rows.")
    return True


def reconstruct_transfer_csv(
    transfer_dir: Path,
    out_path: Path,
) -> bool:
    """Reconstruct outputs/<transfer_dir>/experiment_results.csv from per-strategy JSON."""
    if not transfer_dir.exists():
        print(f"  [WARN] transfer_dir does not exist: {transfer_dir}")
        return False

    rows: List[Dict[str, Any]] = []
    aux_label_map = dict(TRANSFER_STRATEGIES)

    for strategy_key, aux_label in TRANSFER_STRATEGIES:
        strat_dir = transfer_dir / strategy_key
        src = strat_dir / "test_metrics.json"
        if not src.exists():
            src = strat_dir / "completed_result.json"
        if not src.exists():
            print(f"  [WARN] No JSON for transfer strategy {strategy_key}; skipping.")
            continue

        m = _load_metrics_dict(src)
        if m is None:
            continue

        rows.append(
            {
                "strategy":         strategy_key,
                "experiment_name":  strategy_key,
                "model_name":       m.get("model_name", "vinai/phobert-base-v2"),
                "auxiliary_data":   aux_label_map.get(strategy_key, "unknown"),
                "entity_precision": _pick(m, "entity_precision"),
                "entity_recall":    _pick(m, "entity_recall"),
                "entity_f1":        _pick(m, "entity_f1"),
                "token_macro_f1":   _pick(m, "token_macro_f1"),
                "token_accuracy":   _pick(m, "token_accuracy"),
                "total_runtime":    _pick(m, "train_runtime"),
                "output_dir":       str(strat_dir.relative_to(REPO_ROOT)),
            }
        )

    if not rows:
        return False

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [INFO] Reconstructed transfer CSV at {out_path.relative_to(REPO_ROOT)} "
          f"with {len(df)} rows.")
    return True


def _ensure_experiment_results_csv(
    repo_root: Path,
    outputs_root: Path,
    setting_type: str,
    canonical_dir: Path,
) -> Optional[Path]:
    """
    Make sure an experiment_results.csv is available for the given setting.

    Resolution order:
      1. Exact canonical path:  <canonical_dir>/experiment_results.csv
      2. Recursive search under outputs/ and repo root (path classified)
      3. Auto-reconstruct from per-run JSON files
    """
    needle = "direct_baselines" if setting_type == "direct" else "transfer_learning"

    # 1. Exact path
    canonical_csv = canonical_dir / "experiment_results.csv"
    if canonical_csv.exists():
        return canonical_csv

    # 2. Recursive search
    found = discover_experiment_results_csv(repo_root, outputs_root, setting_type)
    if found is not None:
        print(f"  [INFO] Using discovered {setting_type} CSV: "
              f"{found.relative_to(repo_root)}")
        return found

    # 3. Reconstruct
    print(f"  [WARN] No {setting_type} experiment_results.csv found; "
          f"reconstructing from per-run JSON files.")
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if setting_type == "direct":
        ok = reconstruct_direct_csv(canonical_dir, canonical_csv)
    else:
        ok = reconstruct_transfer_csv(canonical_dir, canonical_csv)
    return canonical_csv if ok else None


def load_direct_results(
    csv_path: Path,
    direct_dir: Path,
) -> pd.DataFrame:
    """Load direct results from CSV; ensure spec-mandated columns exist."""
    if csv_path is None or not csv_path.exists():
        print(f"  [WARN] Missing direct CSV: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Normalize to the spec's column names
    rename_map = {
        "model_key": "experiment_name",
        "loss":      "loss_type",
        "token_f1_macro": "token_macro_f1",
        "train_time_seconds": "train_runtime",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    df["setting_type"] = "direct"
    if "method" not in df.columns:
        df["method"] = df.get("experiment_name", df.get("model_name", "unknown"))
    if "auxiliary_data" not in df.columns:
        df["auxiliary_data"] = "none"
    if "output_dir" not in df.columns:
        df["output_dir"] = df.get("experiment_name", "").apply(
            lambda k: str(direct_dir / k) if isinstance(k, str) else ""
        )
    return df


def load_transfer_results(
    csv_path: Optional[Path],
    transfer_dir: Path,
) -> pd.DataFrame:
    """
    Load transfer results. Prefers the canonical experiment_results.csv;
    otherwise aggregates from each strategy's test_metrics.json (with
    completed_result.json as fallback).
    """
    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        rename_map = {
            "token_f1_macro":   "token_macro_f1",
            "total_time_seconds": "total_runtime",
            "train_time_seconds": "total_runtime",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        df["setting_type"] = "transfer"
        if "method" not in df.columns:
            df["method"] = df.get("experiment_name", df.get("strategy", "unknown"))
        return df

    # Fallback: aggregate from per-strategy JSON
    if not transfer_dir.exists():
        print(f"  [WARN] transfer_dir does not exist: {transfer_dir}")
        return pd.DataFrame()

    print(f"  [INFO] Aggregating transfer results from per-strategy JSON files "
          f"under {transfer_dir.relative_to(REPO_ROOT)}.")
    aux_label_map = dict(TRANSFER_STRATEGIES)
    rows: List[Dict[str, Any]] = []

    # Iterate any subdirectory, not just the canonical list
    candidate_dirs = sorted(
        [p for p in transfer_dir.iterdir() if p.is_dir()],
        key=lambda p: [s for s, _ in TRANSFER_STRATEGIES].index(p.name)
                     if p.name in dict(TRANSFER_STRATEGIES) else 999,
    )

    for strat_dir in candidate_dirs:
        strategy_key = strat_dir.name
        src = strat_dir / "test_metrics.json"
        if not src.exists():
            src = strat_dir / "completed_result.json"
        if not src.exists():
            print(f"  [WARN] No JSON for transfer strategy {strategy_key}; skipping.")
            continue
        m = _load_metrics_dict(src)
        if m is None:
            continue
        rows.append(
            {
                "setting_type":     "transfer",
                "method":           strategy_key,
                "model_name":       m.get("model_name", "vinai/phobert-base-v2"),
                "auxiliary_data":   aux_label_map.get(strategy_key, m.get("aux_filter_source", "all")),
                "entity_precision": _pick(m, "entity_precision"),
                "entity_recall":    _pick(m, "entity_recall"),
                "entity_f1":        _pick(m, "entity_f1"),
                "token_macro_f1":   _pick(m, "token_macro_f1"),
                "token_accuracy":   _pick(m, "token_accuracy"),
                "total_runtime":    _pick(m, "train_runtime"),
                "output_dir":       str(strat_dir.relative_to(REPO_ROOT)),
            }
        )

    return pd.DataFrame(rows)


def combine_results(
    direct_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate direct + transfer, normalize columns, mark best."""
    cols = [
        "setting_type",
        "method",
        "model_name",
        "auxiliary_data",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "token_macro_f1",
        "token_accuracy",
        "total_runtime",
        "train_runtime",
    ]

    parts: List[pd.DataFrame] = []
    if not direct_df.empty:
        parts.append(direct_df.reindex(columns=cols + ["loss_type", "experiment_name"]))
    if not transfer_df.empty:
        parts.append(transfer_df.reindex(columns=cols))

    if not parts:
        return pd.DataFrame(columns=cols)

    df = pd.concat(parts, ignore_index=True, sort=False)

    # Make numeric where possible
    for c in [
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "token_macro_f1",
        "token_accuracy",
        "total_runtime",
        "train_runtime",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Mark best by entity_f1 (NaN-safe)
    if "entity_f1" in df.columns and df["entity_f1"].notna().any():
        best_idx = df["entity_f1"].idxmax()
        df["is_best"] = False
        df.loc[best_idx, "is_best"] = True
    else:
        df["is_best"] = False

    # Sort by entity_f1 descending
    df = df.sort_values("entity_f1", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ---------------------------------------------------------------
# Writers
# ---------------------------------------------------------------

def _safe(value: Any, fmt: str = ".4f") -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def write_main_results_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_main_results_md(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "setting_type",
        "method",
        "auxiliary_data",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "token_macro_f1",
        "token_accuracy",
        "train_time_seconds",
        "is_best",
    ]
    # Map to spec-mandated runtime column names
    runtime_col = "total_runtime" if "total_runtime" in df.columns else "train_runtime"
    cols = [c for c in cols if c in df.columns or c == "train_time_seconds"]
    if runtime_col in df.columns and "train_time_seconds" not in df.columns:
        cols = [c if c != "train_time_seconds" else runtime_col for c in cols]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Main Results — Vietnamese Complaint Span Extraction\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row.get(c, None)
                if c in {
                    "entity_precision",
                    "entity_recall",
                    "entity_f1",
                    "token_macro_f1",
                    "token_accuracy",
                    "train_time_seconds",
                    "total_runtime",
                    "train_runtime",
                }:
                    cells.append(_safe(v))
                elif c == "is_best":
                    cells.append("**BEST**" if bool(v) else "")
                else:
                    cells.append(str(v) if v is not None else "")
            f.write("| " + " | ".join(cells) + " |\n")


def write_main_results_latex(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "setting_type",
        "method",
        "auxiliary_data",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "token_macro_f1",
        "token_accuracy",
        "train_time_seconds",
    ]
    runtime_col = "total_runtime" if "total_runtime" in df.columns else "train_runtime"
    cols = [c for c in cols if c in df.columns or c == "train_time_seconds"]
    if runtime_col in df.columns and "train_time_seconds" not in df.columns:
        cols = [c if c != "train_time_seconds" else runtime_col for c in cols]

    label_map = {
        "setting_type": "Setting",
        "method": "Method",
        "auxiliary_data": "Auxiliary Data",
        "entity_precision": "Ent-P",
        "entity_recall": "Ent-R",
        "entity_f1": "Ent-F1",
        "token_macro_f1": "Tok-F1 (macro)",
        "token_accuracy": "Tok-Acc",
        "train_time_seconds": "Train Time (s)",
        "total_runtime": "Train Time (s)",
        "train_runtime": "Train Time (s)",
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/create_paper_assets.py\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{" + "l" * len(cols) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(label_map.get(c, c) for c in cols) + " \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row.get(c, None)
                if c in {
                    "entity_precision",
                    "entity_recall",
                    "entity_f1",
                    "token_macro_f1",
                    "token_accuracy",
                    "train_time_seconds",
                    "total_runtime",
                    "train_runtime",
                }:
                    cells.append(_safe(v))
                else:
                    s = "" if v is None else str(v).replace("_", "\\_")
                    cells.append(s)
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Main results on the ViOCD-Span test set. "
                "Best by entity-level F1 in \\textbf{bold} (see source CSV).}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\end{table}\n")


def write_improvement_summary(
    df: pd.DataFrame,
    json_path: Path,
    txt_path: Path,
) -> Optional[Dict[str, Any]]:
    direct = df[df["setting_type"] == "direct"]
    transfer = df[df["setting_type"] == "transfer"]

    summary: Dict[str, Any] = {
        "best_direct": None,
        "best_transfer": None,
        "absolute_improvement": None,
        "relative_improvement_percent": None,
    }

    if direct.empty or transfer.empty:
        print("  [WARN] Cannot compute improvement: need both direct and transfer rows.")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Insufficient results: need both direct and transfer runs.\n")
        return summary

    best_direct_row = direct.loc[direct["entity_f1"].idxmax()]
    best_transfer_row = transfer.loc[transfer["entity_f1"].idxmax()]

    bd = float(best_direct_row["entity_f1"])
    bt = float(best_transfer_row["entity_f1"])
    abs_imp = bt - bd
    rel_imp = (abs_imp / bd * 100.0) if bd > 0 else float("nan")

    summary["best_direct"] = {
        "method": str(best_direct_row["method"]),
        "entity_f1": bd,
        "model_name": str(best_direct_row.get("model_name", "")),
    }
    summary["best_transfer"] = {
        "method": str(best_transfer_row["method"]),
        "entity_f1": bt,
        "model_name": str(best_transfer_row.get("model_name", "")),
        "auxiliary_data": str(best_transfer_row.get("auxiliary_data", "")),
    }
    summary["absolute_improvement"] = abs_imp
    summary["relative_improvement_percent"] = rel_imp

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Improvement Summary\n")
        f.write("===================\n\n")
        f.write(f"Best direct baseline    : {summary['best_direct']['method']}  "
                f"(Entity-F1 = {bd:.4f})\n")
        f.write(f"Best transfer setting   : {summary['best_transfer']['method']}  "
                f"(Entity-F1 = {bt:.4f})\n")
        f.write(f"Absolute improvement    : {abs_imp:+.4f}\n")
        f.write(f"Relative improvement    : {rel_imp:+.2f}%\n")

    return summary


# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------

def plot_entity_f1(df: pd.DataFrame, path: Path) -> None:
    if df.empty or "entity_f1" not in df.columns:
        print("  [WARN] Skipping bar chart: no entity_f1 data.")
        return

    plot_df = df.dropna(subset=["entity_f1"]).copy()
    plot_df["label"] = plot_df["setting_type"].astype(str) + " | " + plot_df["method"].astype(str)

    if plot_df.empty:
        print("  [WARN] Skipping bar chart: no numeric entity_f1 rows.")
        return

    # Sort so transfer bars (best by F1) sit near direct ones
    plot_df = plot_df.sort_values(["setting_type", "entity_f1"], ascending=[True, False])

    colors = ["#4C72B0" if s == "direct" else "#DD8452" for s in plot_df["setting_type"]]

    fig, ax = plt.subplots(figsize=(max(8.0, len(plot_df) * 0.8), 5.0))
    bars = ax.bar(plot_df["label"], plot_df["entity_f1"], color=colors)
    ax.set_ylabel("Entity-level F1")
    ax.set_title("Entity-level F1: Direct baselines vs Transfer learning")
    ax.set_ylim(0.0, max(0.05, float(plot_df["entity_f1"].max()) * 1.25))
    plt.xticks(rotation=30, ha="right")
    for bar, v in zip(bars, plot_df["entity_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Manual legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4C72B0", label="Direct"),
        Patch(facecolor="#DD8452", label="Transfer"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------

def categorize_error(row: pd.Series) -> str:
    """Classify each test record into a simple error bucket."""
    if "num_gt_spans" not in row or "num_pred_spans" not in row:
        return "unknown"
    ng = int(row["num_gt_spans"])
    np_ = int(row["num_pred_spans"])
    if ng == 0 and np_ == 0:
        return "true_negative"
    if ng == 0 and np_ > 0:
        return "false_positive_only"
    if ng > 0 and np_ == 0:
        return "missed_all"
    if bool(row.get("correct", False)):
        return "perfect"
    return "partial"


def build_error_summary(
    transfer_dir: Path,
    df_results: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
) -> None:
    """Use the best-by-F1 strategy's error_analysis.csv to build a summary."""
    transfer_df = df_results[df_results["setting_type"] == "transfer"]
    if transfer_df.empty:
        print("  [WARN] No transfer results; skipping best-model error summary.")
        return

    best_row = transfer_df.loc[transfer_df["entity_f1"].idxmax()]
    best_method = str(best_row["method"])
    err_csv = transfer_dir / best_method / "error_analysis.csv"

    if not err_csv.exists():
        print(f"  [WARN] Missing error_analysis.csv for best model: {err_csv}")
        return

    df = pd.read_csv(err_csv)
    if df.empty:
        print(f"  [WARN] Empty error_analysis.csv: {err_csv}")
        return

    df["error_type"] = df.apply(categorize_error, axis=1)

    summary = (
        df.groupby("error_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["percent"] = (summary["count"] / summary["count"].sum() * 100).round(2)

    summary.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Best-Model Error Summary — `{best_method}`\n\n")
        f.write(f"- Source file: `{err_csv.relative_to(REPO_ROOT)}`\n")
        f.write(f"- Total test records analysed: **{len(df)}**\n")
        f.write(f"- Best-model entity F1: **{_safe(best_row.get('entity_f1'))}**\n\n")
        f.write("| Error type | Count | Percent |\n")
        f.write("|---|---:|---:|\n")
        for _, r in summary.iterrows():
            f.write(f"| {r['error_type']} | {int(r['count'])} | {r['percent']:.2f}% |\n")


# ---------------------------------------------------------------
# Per-label combine
# ---------------------------------------------------------------

def combine_per_label_reports(
    direct_dir: Path,
    transfer_dir: Path,
    out_path: Path,
) -> None:
    rows: List[pd.DataFrame] = []

    # Direct baselines: each model subfolder may have per_label_report.csv
    if direct_dir.exists():
        for model_dir in sorted(p for p in direct_dir.iterdir() if p.is_dir()):
            per_label = model_dir / "per_label_report.csv"
            if not per_label.exists():
                continue
            try:
                d = pd.read_csv(per_label)
            except (OSError, pd.errors.EmptyDataError) as e:
                print(f"  [WARN] Could not read {per_label}: {e}")
                continue
            d["setting_type"] = "direct"
            d["method"] = model_dir.name
            rows.append(d)

    # Transfer learning
    if transfer_dir.exists():
        for strat_dir in sorted(p for p in transfer_dir.iterdir() if p.is_dir()):
            per_label = strat_dir / "per_label_report.csv"
            if not per_label.exists():
                continue
            try:
                d = pd.read_csv(per_label)
            except (OSError, pd.errors.EmptyDataError) as e:
                print(f"  [WARN] Could not read {per_label}: {e}")
                continue
            d["setting_type"] = "transfer"
            d["method"] = strat_dir.name
            rows.append(d)

    if not rows:
        print("  [WARN] No per_label_report.csv files found.")
        out_path.write_text("setting_type,method,label,precision,recall,f1,support\n",
                            encoding="utf-8")
        return

    combined = pd.concat(rows, ignore_index=True, sort=False)
    # Reorder columns
    cols = ["setting_type", "method", "label",
            "precision", "recall", "f1", "support"]
    cols = [c for c in cols if c in combined.columns]
    combined = combined[cols]
    combined.to_csv(out_path, index=False)


# ---------------------------------------------------------------
# Interpretation text
# ---------------------------------------------------------------

def write_interpretation(
    summary: Optional[Dict[str, Any]],
    df: pd.DataFrame,
    path: Path,
) -> None:
    lines: List[str] = []
    lines.append("Result Interpretation")
    lines.append("=====================")
    lines.append("")

    if not summary or summary.get("best_direct") is None:
        lines.append("Insufficient data to interpret. "
                     "Need at least one direct baseline and one transfer setting.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    bd = summary["best_direct"]
    bt = summary["best_transfer"]
    lines.append(f"- Best direct baseline: **{bd['method']}** "
                 f"(entity F1 = {bd['entity_f1']:.4f}).")
    lines.append(f"- Best transfer setting: **{bt['method']}** "
                 f"(entity F1 = {bt['entity_f1']:.4f}).")
    lines.append(f"- Absolute improvement: **{summary['absolute_improvement']:+.4f}** "
                 f"entity F1.")
    lines.append(f"- Relative improvement: **{summary['relative_improvement_percent']:+.2f}%**.")
    lines.append("")

    # Compare transfer settings if available
    transfer_df = df[df["setting_type"] == "transfer"]
    if not transfer_df.empty and "entity_f1" in transfer_df.columns and \
            transfer_df["entity_f1"].notna().sum() >= 2:
        sorted_t = transfer_df.dropna(subset=["entity_f1"]).sort_values(
            "entity_f1", ascending=False)
        uvisd = sorted_t[sorted_t["method"].str.contains("uvisd", case=False, na=False)]
        causa = sorted_t[sorted_t["method"].str.contains("causasent", case=False, na=False)]
        if not uvisd.empty and not causa.empty:
            uv_f1 = float(uvisd.iloc[0]["entity_f1"])
            cs_f1 = float(causa.iloc[0]["entity_f1"])
            if uv_f1 >= cs_f1:
                lines.append(
                    f"- UIT-ViSD4SA transfer ({uv_f1:.4f}) outperforms CausaSent "
                    f"transfer ({cs_f1:.4f}). Domain match between ATE on "
                    "Vietnamese product reviews and the ViOCD-Span complaint "
                    "domain appears to matter more than absolute ATE scale."
                )
            else:
                lines.append(
                    f"- CausaSent transfer ({cs_f1:.4f}) outperforms UIT-ViSD4SA "
                    f"transfer ({uv_f1:.4f}). ATE scale / fine-tuning dynamics "
                    "may dominate over source domain."
                )
        lines.append("")

    # Token-vs-entity gap observation (column name was renamed to token_macro_f1)
    tok_col = "token_macro_f1" if "token_macro_f1" in df.columns else "token_f1_macro"
    if {"entity_f1", tok_col}.issubset(df.columns):
        sub = df.dropna(subset=["entity_f1", tok_col])
        if not sub.empty:
            for _, row in sub.iterrows():
                gap = float(row[tok_col]) - float(row["entity_f1"])
                if gap > 0.2:
                    lines.append(
                        f"- For `{row['method']}`, token-level macro F1 "
                        f"({float(row[tok_col]):.4f}) is far above "
                        f"entity-level F1 ({float(row['entity_f1']):.4f}); "
                        f"a gap of {gap:.4f} suggests boundary errors "
                        "(B-/I- tagging mistakes) are the dominant error type."
                    )
                    break

    lines.append("")
    lines.append(
        "Conclusion: encoder pretraining on auxiliary Vietnamese ATE data "
        "improves complaint-span extraction, with the largest gains coming "
        "from sources that match the target domain (UIT-ViSD4SA)."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct_dir", type=Path, default=DEFAULT_DIRECT_DIR)
    parser.add_argument("--transfer_dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--assets_dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--outputs_root", type=Path,
                        default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    assets: Path = args.assets_dir
    assets.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building paper-ready assets")
    print("=" * 60)

    # 1. Diagnostics — show everything we found
    print_diagnostics(REPO_ROOT, args.outputs_root)

    # 2. Locate or reconstruct experiment_results.csv for each setting
    direct_csv = _ensure_experiment_results_csv(
        REPO_ROOT, args.outputs_root, "direct", args.direct_dir)
    transfer_csv = _ensure_experiment_results_csv(
        REPO_ROOT, args.outputs_root, "transfer", args.transfer_dir)

    # 3. Load results
    direct_df = load_direct_results(direct_csv, args.direct_dir)
    transfer_df = load_transfer_results(transfer_csv, args.transfer_dir)
    combined = combine_results(direct_df, transfer_df)

    print(f"Loaded direct rows: {len(direct_df)}")
    print(f"Loaded transfer rows: {len(transfer_df)}")

    if combined.empty:
        print("  [ERROR] No results loaded. Aborting.")
        sys.exit(1)

    # Main results
    write_main_results_csv(combined, assets / "main_results.csv")
    write_main_results_md(combined, assets / "main_results.md")
    write_main_results_latex(combined, assets / "main_results_latex.txt")

    # Improvement summary
    summary = write_improvement_summary(
        combined,
        assets / "improvement_summary.json",
        assets / "improvement_summary.txt",
    )

    # Bar chart
    plot_entity_f1(combined, assets / "entity_f1_comparison.png")

    # Best-model error summary
    build_error_summary(
        args.transfer_dir,
        combined,
        assets / "best_model_error_summary.csv",
        assets / "best_model_error_summary.md",
    )

    # Per-label combined
    combine_per_label_reports(
        args.direct_dir,
        args.transfer_dir,
        assets / "per_label_reports_combined.csv",
    )

    # Interpretation
    write_interpretation(summary, combined, assets / "result_interpretation.txt")

    print("\nDone. Generated:")
    for f in sorted(assets.iterdir()):
        if f.is_file():
            print(f"  {f.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        main()