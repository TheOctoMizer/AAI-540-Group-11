"""
01_data_overview.py — Dataset Health Check
===========================================
Big-picture audit of the CICIDS2017 dataset:
  - Per-file row counts, sizes, feature counts
  - Global missing value audit + heatmap
  - Infinite value audit + bar chart
  - Full feature summary statistics table (CSV)

Outputs → nids_eda/outputs/
  data_summary_table.csv
  missing_values_heatmap.png
  inf_values_barplot.png
  per_file_stats.png

Usage:
    python 01_data_overview.py [--data-dir ../nids_train/nids_dataset] [--output-dir ./outputs]
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Palette ──────────────────────────────────────────────────────────────────
ACCENT   = "#4C72B0"
WARN     = "#DD8452"
DANGER   = "#C44E52"
BG       = "#F8F9FA"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG})

# ─── Dataset config ───────────────────────────────────────────────────────────
DATASET_FILES = [
    ("Monday",    "Monday-WorkingHours.pcap_ISCX.csv"),
    ("Tuesday",   "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("Wednesday", "Wednesday-workingHours.pcap_ISCX.csv"),
    ("Thursday-AM","Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("Thursday-PM","Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("Friday-AM", "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("Friday-DDoS","Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    ("Friday-Scan","Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
]

DROP_COLS = {"Flow ID", "Source IP", "Source Port", "Destination IP",
             "Destination Port", "Protocol", "Timestamp", "Label"}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in DROP_COLS]


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Per-file stats ──────────────────────────────────────────────────
    print("=" * 70)
    print("01 · DATA OVERVIEW")
    print("=" * 70)

    file_rows, file_mb, day_labels = [], [], []
    all_dfs = []

    for day, fname in DATASET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  ⚠  {fname} not found — skipping")
            continue
        mb = fpath.stat().st_size / 1_048_576
        df = load_csv(fpath)
        file_rows.append(len(df))
        file_mb.append(mb)
        day_labels.append(day)
        all_dfs.append(df)
        print(f"  {day:<12} {len(df):>10,} rows  |  {mb:>7.1f} MB  |  {len(df.columns)} cols")

    # ── 2. Combine ─────────────────────────────────────────────────────────
    print("\nCombining datasets …")
    combined = pd.concat(all_dfs, ignore_index=True)
    fcols = feature_cols(combined)
    feat_df = combined[fcols].copy()
    print(f"  Total rows : {len(combined):,}")
    print(f"  Features   : {len(fcols)}")

    # ── 3. Per-file stats chart ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("CICIDS2017 — Per-File Dataset Sizes", fontsize=14, fontweight="bold")

    axes[0].barh(day_labels, file_rows, color=ACCENT, edgecolor="white")
    axes[0].set_xlabel("Number of Rows")
    axes[0].set_title("Row Count per File")
    for bar, val in zip(axes[0].patches, file_rows):
        axes[0].text(bar.get_width() + max(file_rows) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=8)

    axes[1].barh(day_labels, file_mb, color=WARN, edgecolor="white")
    axes[1].set_xlabel("File Size (MB)")
    axes[1].set_title("File Size per CSV")
    for bar, val in zip(axes[1].patches, file_mb):
        axes[1].text(bar.get_width() + max(file_mb) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "per_file_stats.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ per_file_stats.png")

    # ── 4. Missing values ──────────────────────────────────────────────────
    print("\nAuditing missing & infinite values …")
    feat_clean = feat_df.replace([np.inf, -np.inf], np.nan)
    miss_pct   = feat_clean.isnull().mean() * 100               # per feature
    miss_cols  = miss_pct[miss_pct > 0].sort_values(ascending=False)

    if len(miss_cols) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(miss_cols) * 0.7), 5), facecolor=BG)
        miss_cols.plot(kind="bar", ax=ax, color=WARN, edgecolor="white")
        ax.set_title("Features with Missing Values (% of rows)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Missing %")
        ax.set_xlabel("Feature")
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        for bar, pct in zip(ax.patches, miss_cols.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{pct:.2f}%", ha="center", fontsize=7)
        plt.tight_layout()
        fig.savefig(out_dir / "missing_values_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ missing_values_heatmap.png  ({len(miss_cols)} features with nulls)")
    else:
        print("  ✓ No missing values detected")

    # ── Heatmap of completeness (rows × features sampled) ─────────────────
    sample = feat_clean.sample(n=min(5000, len(feat_clean)), random_state=42)
    null_map = sample.isnull().any(axis=0)
    null_features = null_map[null_map].index.tolist()
    if null_features:
        hmdf = sample[null_features].isnull()
        fig, ax = plt.subplots(figsize=(min(20, len(null_features) * 0.8 + 2), 5), facecolor=BG)
        sns.heatmap(hmdf.T, cbar=False, cmap=["#4C72B0", "#C44E52"],
                    yticklabels=True, xticklabels=False, ax=ax)
        ax.set_title("Missing Value Presence (5k-row sample · red = missing)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample rows")
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        fig.savefig(out_dir / "missing_heatmap_detail.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── 5. Infinite values ─────────────────────────────────────────────────
    inf_counts = {}
    for col in fcols:
        n_inf = np.isinf(feat_df[col].values).sum()
        if n_inf > 0:
            inf_counts[col] = n_inf

    if inf_counts:
        inf_series = pd.Series(inf_counts).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(inf_series) * 0.7), 5), facecolor=BG)
        inf_series.plot(kind="bar", ax=ax, color=DANGER, edgecolor="white")
        ax.set_title("Features with Infinite Values (absolute count)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count of Inf rows")
        ax.set_xlabel("Feature")
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        for bar, val in zip(ax.patches, inf_series.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(inf_series) * 0.01,
                    f"{val:,}", ha="center", fontsize=7)
        plt.tight_layout()
        fig.savefig(out_dir / "inf_values_barplot.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ inf_values_barplot.png  ({len(inf_counts)} features with Inf)")
    else:
        print("  ✓ No infinite values detected")

    # ── 6. Feature summary stats CSV ──────────────────────────────────────
    print("\nComputing feature summary statistics …")
    rows = []
    for col in fcols:
        series = feat_clean[col].dropna()
        rows.append({
            "feature":     col,
            "dtype":       str(feat_df[col].dtype),
            "count":       len(series),
            "mean":        series.mean(),
            "std":         series.std(),
            "min":         series.min(),
            "p25":         series.quantile(0.25),
            "p50":         series.quantile(0.50),
            "p75":         series.quantile(0.75),
            "p95":         series.quantile(0.95),
            "p99":         series.quantile(0.99),
            "max":         series.max(),
            "missing_pct": feat_clean[col].isnull().mean() * 100,
            "inf_count":   inf_counts.get(col, 0),
            "zero_pct":    (series == 0).mean() * 100,
        })
    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "data_summary_table.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"  ✓ data_summary_table.csv  ({len(summary_df)} features)")

    # ── 7. Console report ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"{'FEATURE':<40} {'MEAN':>12} {'STD':>12} {'INF':>8} {'MISS%':>7}")
    print("─" * 70)
    for _, row in summary_df.head(20).iterrows():
        print(f"{row.feature:<40} {row['mean']:>12.4f} {row['std']:>12.4f} "
              f"{int(row.inf_count):>8,} {row.missing_pct:>6.2f}%")
    print("─" * 70)
    print(f"\n✓ All outputs saved to: {out_dir}")
    print(f"  Total dataset: {len(combined):,} rows × {len(fcols)} features")
    print(f"  Files missing : {miss_cols.shape[0]} features with nulls")
    print(f"  Files with Inf: {len(inf_counts)} features with ∞")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="01 — Data Overview")
    parser.add_argument("--data-dir",   default="../nids_train/nids_dataset")
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir))
