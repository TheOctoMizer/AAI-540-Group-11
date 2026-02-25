"""
04_correlation_analysis.py — Feature Correlation & Redundancy
==============================================================
Identify collinear/redundant features to guide dimensionality reduction.

Outputs → nids_eda/outputs/
  full_correlation_heatmap.png        (all 79 features)
  clustered_correlation_heatmap.png   (hierarchical clustering)
  high_correlation_pairs.csv          (|r| > threshold pairs)
  correlation_distribution.png        (histogram of all pairwise |r|)

Usage:
    python 04_correlation_analysis.py [--data-dir ..] [--output-dir ./outputs] [--sample 0.05] [--threshold 0.90]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BG = "#F8F9FA"
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG})

DATASET_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]
DROP_COLS = {"Flow ID", "Source IP", "Source Port", "Destination IP",
             "Destination Port", "Protocol", "Timestamp", "Label"}


def load_sample(data_dir: Path, frac: float) -> pd.DataFrame:
    frames = []
    for fname in DATASET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df.sample(frac=frac, random_state=42))
    combined = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in combined.columns if c not in DROP_COLS]
    return combined[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()


def main(data_dir: Path, out_dir: Path, frac: float, threshold: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("04 · CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"  Sample fraction : {frac:.1%}")
    print(f"  High-corr thresh: |r| > {threshold}")

    df = load_sample(data_dir, frac)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} features")

    # ── 1. Pearson correlation matrix ──────────────────────────────────────
    print("\nComputing Pearson correlation matrix …")
    corr = df.corr(method="pearson")

    # Full 79×79 heatmap
    feat_count = len(corr)
    font_size  = max(3.5, min(7, 120 / feat_count))  # scale with features
    fig_size   = max(18, feat_count * 0.28)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85), facecolor=BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0, ax=ax,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"},
                xticklabels=True, yticklabels=True)
    ax.set_title(f"Feature Correlation Matrix ({feat_count} features) — Lower Triangle",
                 fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=font_size, rotation=90)
    ax.tick_params(axis="y", labelsize=font_size, rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "full_correlation_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ full_correlation_heatmap.png")

    # ── 2. Clustered correlation heatmap ──────────────────────────────────
    print("  Generating clustered heatmap …")
    try:
        cg = sns.clustermap(corr, cmap="coolwarm", vmin=-1, vmax=1,
                            figsize=(fig_size, fig_size * 0.85),
                            xticklabels=True, yticklabels=True,
                            dendrogram_ratio=0.12,
                            cbar_pos=(0.02, 0.8, 0.03, 0.15),
                            linewidths=0)
        cg.ax_heatmap.tick_params(axis="x", labelsize=font_size, rotation=90)
        cg.ax_heatmap.tick_params(axis="y", labelsize=font_size, rotation=0)
        cg.fig.suptitle("Clustered Correlation Matrix (Hierarchical)", fontsize=12,
                        fontweight="bold", y=1.01)
        cg.savefig(out_dir / "clustered_correlation_heatmap.png", dpi=140, bbox_inches="tight")
        plt.close("all")
        print("  ✓ clustered_correlation_heatmap.png")
    except Exception as e:
        print(f"  ⚠  Clustered heatmap failed: {e}")

    # ── 3. Distribution of pairwise |r| ───────────────────────────────────
    upper_tri = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    abs_r = upper_tri.stack().abs()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.hist(abs_r.values, bins=80, color="#4C72B0", edgecolor="none", alpha=0.8)
    ax.axvline(threshold, color="#C44E52", lw=2, linestyle="--",
               label=f"Threshold |r| = {threshold}")
    ax.axvline(0.5, color="#DD8452", lw=1.5, linestyle=":",
               label="|r| = 0.5 (moderate)")
    pct_high = (abs_r >= threshold).mean() * 100
    ax.set_xlabel("Absolute Pearson Correlation |r|", fontsize=11)
    ax.set_ylabel("Number of Feature Pairs")
    ax.set_title(f"Distribution of Pairwise |r| — {pct_high:.1f}% pairs above {threshold}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "correlation_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ correlation_distribution.png")

    # ── 4. High-correlation pairs CSV ─────────────────────────────────────
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j],
                              "pearson_r": round(r, 6), "abs_r": round(abs(r), 6)})
    pairs_df = pd.DataFrame(pairs).sort_values("abs_r", ascending=False)
    pairs_df.to_csv(out_dir / "high_correlation_pairs.csv", index=False)
    print(f"  ✓ high_correlation_pairs.csv  ({len(pairs_df)} pairs with |r| ≥ {threshold})")

    # Console summary
    print(f"\n  Top-15 most correlated pairs:")
    print(f"  {'Feature A':<38} {'Feature B':<38} {'r':>8}")
    print("  " + "─" * 86)
    for _, row in pairs_df.head(15).iterrows():
        print(f"  {row.feature_a:<38} {row.feature_b:<38} {row.pearson_r:>8.4f}")

    # ── 5. Suggest features to drop ───────────────────────────────────────
    to_drop = set()
    for _, row in pairs_df.iterrows():
        # Keep feature_a, mark feature_b for potential drop (greedy)
        if row.feature_b not in to_drop:
            to_drop.add(row.feature_b)

    print(f"\n  Features that could be dropped (greedy high-corr removal):")
    print(f"  {len(to_drop)} / {feat_count} features flagged")
    for f in sorted(to_drop)[:20]:
        print(f"    • {f}")
    if len(to_drop) > 20:
        print(f"    … and {len(to_drop) - 20} more (see high_correlation_pairs.csv)")

    print(f"\n✓ Done. Outputs → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="04 — Correlation Analysis")
    parser.add_argument("--data-dir",   default="../nids_train/nids_dataset")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--sample",     type=float, default=0.05)
    parser.add_argument("--threshold",  type=float, default=0.90,
                        help="Absolute correlation threshold for high-corr pairs (default 0.90)")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir), args.sample, args.threshold)
