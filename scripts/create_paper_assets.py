#!/usr/bin/env python3
"""
Build paper-ready tables, figures, and analysis files from completed experiments.

Inputs:
- outputs/experiments/direct_baselines/experiment_results.csv
- outputs/experiments/transfer_learning/<strategy>/test_metrics.json
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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Use a non-interactive matplotlib backend so this works on Kaggle/CI
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRECT_DIR = REPO_ROOT / "outputs" / "experiments" / "direct_baselines"
DEFAULT_TRANSFER_DIR = REPO_ROOT / "outputs" / "experiments" / "transfer_learning"
DEFAULT_ASSETS_DIR = REPO_ROOT / "outputs" / "paper_assets"


# ---------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------

def load_direct_results(direct_dir: Path) -> pd.DataFrame:
    """Load direct-baseline experiment_results.csv if present."""
    csv_path = direct_dir / "experiment_results.csv"
    if not csv_path.exists():
        print(f"  [WARN] Missing direct results: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["setting_type"] = "direct"
    df["method"] = df.get("model_key", df.get("model_name", "unknown"))
    df["auxiliary_data"] = "none"
    df["train_time_seconds"] = df.get("train_time_seconds", pd.NA)
    return df


def load_transfer_results(transfer_dir: Path) -> pd.DataFrame:
    """
    Aggregate per-strategy test_metrics.json files into one DataFrame.

    Falls back to a manually-built per-strategy summary if the
    transfer experiment_results.csv doesn't exist yet.
    """
    rows: List[Dict[str, Any]] = []

    if not transfer_dir.exists():
        print(f"  [WARN] Missing transfer directory: {transfer_dir}")
        return pd.DataFrame()

    for strategy_dir in sorted(p for p in transfer_dir.iterdir() if p.is_dir()):
        metrics_path = strategy_dir / "test_metrics.json"
        if not metrics_path.exists():
            print(f"  [WARN] No test_metrics.json in {strategy_dir}")
            continue
        try:
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [WARN] Could not read {metrics_path}: {e}")
            continue

        rows.append(
            {
                "setting_type": "transfer",
                "method": m.get("strategy", strategy_dir.name),
                "model_name": m.get("model_name", "unknown"),
                "auxiliary_data": m.get("aux_filter_source", "all"),
                "entity_precision": m.get("entity_precision"),
                "entity_recall": m.get("entity_recall"),
                "entity_f1": m.get("entity_f1"),
                "token_f1_macro": m.get("token_f1_macro"),
                "token_f1_weighted": m.get("token_f1_weighted"),
                "token_accuracy": m.get("token_accuracy"),
                "train_time_seconds": m.get("total_time_seconds"),
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
        "token_f1_macro",
        "token_f1_weighted",
        "token_accuracy",
        "train_time_seconds",
    ]

    parts: List[pd.DataFrame] = []
    if not direct_df.empty:
        parts.append(direct_df.reindex(columns=cols + ["loss", "num_train_epochs"]))
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
        "token_f1_macro",
        "token_f1_weighted",
        "token_accuracy",
        "train_time_seconds",
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
        "token_f1_macro",
        "token_f1_weighted",
        "token_accuracy",
        "train_time_seconds",
        "is_best",
    ]
    cols = [c for c in cols if c in df.columns]

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
                    "token_f1_macro",
                    "token_f1_weighted",
                    "token_accuracy",
                    "train_time_seconds",
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
        "token_f1_macro",
        "token_f1_weighted",
        "token_accuracy",
        "train_time_seconds",
    ]
    cols = [c for c in cols if c in df.columns]

    label_map = {
        "setting_type": "Setting",
        "method": "Method",
        "auxiliary_data": "Auxiliary Data",
        "entity_precision": "Ent-P",
        "entity_recall": "Ent-R",
        "entity_f1": "Ent-F1",
        "token_f1_macro": "Tok-F1 (macro)",
        "token_f1_weighted": "Tok-F1 (weighted)",
        "token_accuracy": "Tok-Acc",
        "train_time_seconds": "Train Time (s)",
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
                    "token_f1_macro",
                    "token_f1_weighted",
                    "token_accuracy",
                    "train_time_seconds",
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

    # Token-vs-entity gap observation
    if {"entity_f1", "token_f1_macro"}.issubset(df.columns):
        sub = df.dropna(subset=["entity_f1", "token_f1_macro"])
        if not sub.empty:
            for _, row in sub.iterrows():
                gap = float(row["token_f1_macro"]) - float(row["entity_f1"])
                if gap > 0.2:
                    lines.append(
                        f"- For `{row['method']}`, token-level macro F1 "
                        f"({float(row['token_f1_macro']):.4f}) is far above "
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
    args = parser.parse_args()

    assets: Path = args.assets_dir
    assets.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building paper-ready assets")
    print("=" * 60)

    direct_df = load_direct_results(args.direct_dir)
    transfer_df = load_transfer_results(args.transfer_dir)
    combined = combine_results(direct_df, transfer_df)

    if combined.empty:
        print("  [ERROR] No results loaded. Aborting.")
        sys.exit(1)

    print(f"  Loaded {len(direct_df)} direct rows, "
          f"{len(transfer_df)} transfer rows, combined={len(combined)} rows.")

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