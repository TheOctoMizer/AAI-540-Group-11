"""
03_feature_analysis.py — Feature Distribution Analysis
=======================================================
Understand individual feature distributions and discriminative power.

Outputs → nids_eda/outputs/
  top_variance_histograms.png   (top-20 by variance with KDE)
  boxplots_top_features.png     (top-10 by separation power, per class group)
  violin_flow_features.png      (flow duration / packet-length / IAT violins)
  zero_ratio_per_feature.png    (sparsity — % zeros per feature)
  feature_groups_summary.png    (grouped heatmap of feature families)

Usage:
    python 03_feature_analysis.py [--data-dir ..] [--output-dir ./outputs] [--sample 0.1]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

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
             "Destination Port", "Protocol", "Timestamp"}

# Named feature groups for grouped heatmap
FEATURE_GROUPS = {
    "Flow Duration":   ["Flow Duration"],
    "Fwd Pkt Len":     ["Fwd Packet Length Max", "Fwd Packet Length Min",
                        "Fwd Packet Length Mean", "Fwd Packet Length Std"],
    "Bwd Pkt Len":     ["Bwd Packet Length Max", "Bwd Packet Length Min",
                        "Bwd Packet Length Mean", "Bwd Packet Length Std"],
    "Flow Rates":      ["Flow Bytes/s", "Flow Packets/s",
                        "Fwd Packets/s", "Bwd Packets/s"],
    "Flow IAT":        ["Flow IAT Mean", "Flow IAT Std",
                        "Flow IAT Max", "Flow IAT Min"],
    "Fwd IAT":         ["Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
                        "Fwd IAT Max", "Fwd IAT Min"],
    "Bwd IAT":         ["Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
                        "Bwd IAT Max", "Bwd IAT Min"],
    "TCP Flags":       ["FIN Flag Count", "SYN Flag Count", "RST Flag Count",
                        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
                        "CWE Flag Count", "ECE Flag Count",
                        "Fwd PSH Flags", "Bwd PSH Flags",
                        "Fwd URG Flags", "Bwd URG Flags"],
    "Pkt Stats":       ["Min Packet Length", "Max Packet Length",
                        "Packet Length Mean", "Packet Length Std",
                        "Packet Length Variance", "Average Packet Size"],
    "Window/Bulk":     ["Init_Win_bytes_forward", "Init_Win_bytes_backward",
                        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
                        "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
                        "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"],
    "Subflow":         ["Subflow Fwd Packets", "Subflow Fwd Bytes",
                        "Subflow Bwd Packets", "Subflow Bwd Bytes"],
    "Active/Idle":     ["Active Mean", "Active Std", "Active Max", "Active Min",
                        "Idle Mean", "Idle Std", "Idle Max", "Idle Min"],
}

# Simplified class groups for visualization
CLASS_GROUPS = {
    "BENIGN":  "BENIGN",
    "DoS":     ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"],
    "DDoS":    ["DDoS"],
    "Scan":    ["PortScan"],
    "BFA":     ["FTP-Patator", "SSH-Patator"],
    "Web":     ["Bot"],   # reuse slot for Botnet for simplicity
    "Botnet":  ["Bot"],
}


def load_sample(data_dir: Path, sample_frac: float) -> pd.DataFrame:
    frames = []
    for fname in DATASET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].str.strip()
        frames.append(df.sample(frac=sample_frac, random_state=42))
    combined = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in combined.columns if c not in DROP_COLS]
    combined = combined[feat_cols].replace([np.inf, -np.inf], np.nan)
    return combined


def map_class_group(label: str) -> str:
    label = str(label).strip()
    if label == "BENIGN":
        return "BENIGN"
    if "dos" in label.lower() or "heartbleed" in label.lower() or "slowhttptest" in label.lower():
        return "DoS"
    if "ddos" in label.lower():
        return "DDoS"
    if "portscan" in label.lower():
        return "PortScan"
    if "patator" in label.lower():
        return "Brute Force"
    if "web attack" in label.lower():
        return "Web Attack"
    if "bot" in label.lower():
        return "Botnet"
    if "infiltration" in label.lower():
        return "Infiltration"
    return "Other"


def main(data_dir: Path, out_dir: Path, sample_frac: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("03 · FEATURE ANALYSIS")
    print("=" * 70)
    print(f"  Sample fraction: {sample_frac:.1%}")

    # ── Load ───────────────────────────────────────────────────────────────
    df = load_sample(data_dir, sample_frac)
    feat_cols = [c for c in df.columns if c != "Label"]
    print(f"  Loaded {len(df):,} rows × {len(feat_cols)} features")

    # ── 1. Top-20 by variance — histograms with KDE ────────────────────────
    print("\nComputing feature variances …")
    feat_num = df[feat_cols].dropna()
    variances = feat_num.var().sort_values(ascending=False)
    top20 = variances.head(20).index.tolist()

    fig, axes = plt.subplots(4, 5, figsize=(22, 16), facecolor=BG)
    fig.suptitle(f"Top-20 Features by Variance (sample={sample_frac:.0%})",
                 fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for idx, feat in enumerate(top20):
        ax = axes[idx]
        ax.set_facecolor(BG)
        data = feat_num[feat].dropna()
        # Clip to p1–p99 to suppress outlier dominance on axis
        lo, hi = data.quantile(0.01), data.quantile(0.99)
        clipped = data.clip(lo, hi)
        ax.hist(clipped, bins=60, color="#4C72B0", alpha=0.7, edgecolor="none", density=True)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(clipped.sample(min(5000, len(clipped)), random_state=42))
            xgrid = np.linspace(clipped.min(), clipped.max(), 200)
            ax.plot(xgrid, kde(xgrid), color="#C44E52", lw=1.5)
        except Exception:
            pass
        ax.set_title(f"{feat[:28]}\nvar={variances[feat]:.2e}", fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=6)
        skw = skew(clipped.values)
        krt = kurtosis(clipped.values)
        ax.text(0.97, 0.97, f"skew={skw:.1f}\nkurt={krt:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    plt.tight_layout()
    fig.savefig(out_dir / "top_variance_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ top_variance_histograms.png")

    # ── 2. Box plots — top 10 features, per class group ───────────────────
    df["class_group"] = df["Label"].map(map_class_group)
    groups_order = ["BENIGN", "DoS", "DDoS", "PortScan", "Brute Force",
                    "Web Attack", "Botnet", "Infiltration"]
    groups_present = [g for g in groups_order if g in df["class_group"].unique()]
    palette = sns.color_palette("Set2", len(groups_present))

    top10 = variances.head(10).index.tolist()
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), facecolor=BG)
    fig.suptitle("Top-10 Features (by Variance) — Distribution per Attack Group",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for idx, feat in enumerate(top10):
        ax = axes[idx]
        ax.set_facecolor(BG)
        plot_df = df[["class_group", feat]].dropna()
        lo = plot_df[feat].quantile(0.01)
        hi = plot_df[feat].quantile(0.99)
        plot_df = plot_df[plot_df[feat].between(lo, hi)]
        sns.boxplot(data=plot_df, x="class_group", y=feat, order=groups_present,
                    palette=palette, width=0.6, flierprops={"marker": ".", "ms": 2},
                    ax=ax, linewidth=0.8)
        ax.set_title(feat[:35], fontsize=8, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=35)
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "boxplots_top_features.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ boxplots_top_features.png")

    # ── 3. Violin plots — key flow features ───────────────────────────────
    violin_features = [f for f in [
        "Flow Duration", "Fwd Packet Length Mean",
        "Bwd Packet Length Mean", "Flow IAT Mean", "Fwd IAT Mean",
        "Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    ] if f in df.columns]

    if violin_features:
        n_cols = 4
        n_rows = (len(violin_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4 + 1), facecolor=BG)
        fig.suptitle("Violin Plots — Key Flow Features per Attack Group",
                     fontsize=13, fontweight="bold")
        axes = axes.flatten()

        for idx, feat in enumerate(violin_features):
            ax = axes[idx]
            ax.set_facecolor(BG)
            plot_df = df[["class_group", feat]].dropna()
            lo = plot_df[feat].quantile(0.02)
            hi = plot_df[feat].quantile(0.98)
            plot_df = plot_df[plot_df[feat].between(lo, hi)]
            try:
                sns.violinplot(data=plot_df, x="class_group", y=feat,
                               order=groups_present, palette=palette,
                               inner="quartile", cut=0, linewidth=0.8, ax=ax)
            except Exception:
                sns.boxplot(data=plot_df, x="class_group", y=feat,
                            order=groups_present, palette=palette, ax=ax)
            ax.set_title(feat[:35], fontsize=9, pad=4)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=7, rotation=35)
            ax.tick_params(axis="y", labelsize=7)

        # Hide unused axes
        for j in range(len(violin_features), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fig.savefig(out_dir / "violin_flow_features.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ violin_flow_features.png")

    # ── 4. Zero ratio per feature ──────────────────────────────────────────
    zero_pct = (feat_num == 0).mean() * 100
    zero_pct_sorted = zero_pct.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(22, 6), facecolor=BG)
    ax.set_facecolor(BG)
    colors = ["#C44E52" if p > 80 else "#DD8452" if p > 30 else "#4C72B0"
              for p in zero_pct_sorted.values]
    ax.bar(range(len(zero_pct_sorted)), zero_pct_sorted.values, color=colors, edgecolor="none")
    ax.set_xticks(range(len(zero_pct_sorted)))
    ax.set_xticklabels([f[:20] for f in zero_pct_sorted.index], rotation=90, fontsize=5.5)
    ax.set_ylabel("Percentage of Zero Values (%)")
    ax.set_title("Feature Sparsity — Percentage of Zero Values per Feature",
                 fontsize=13, fontweight="bold")
    ax.axhline(80, color="#C44E52", lw=1.5, linestyle="--", label=">80% zeros (very sparse)")
    ax.axhline(30, color="#DD8452", lw=1.5, linestyle="--", label=">30% zeros")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "zero_ratio_per_feature.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ zero_ratio_per_feature.png")

    # ── 5. Feature group median heatmap ───────────────────────────────────
    group_medians = {}
    for grp_name, feats in FEATURE_GROUPS.items():
        valid = [f for f in feats if f in feat_num.columns]
        if valid:
            group_medians[grp_name] = feat_num[valid].median()

    if group_medians:
        # Per class-group × feature-group — median of medians (normalized)
        class_group_median = {}
        for cg in groups_present:
            sub = df[df["class_group"] == cg]
            sub_num = sub[[c for c in feat_cols if c in feat_num.columns]].dropna()
            row = {}
            for grp_name, feats in FEATURE_GROUPS.items():
                valid = [f for f in feats if f in sub_num.columns]
                if valid:
                    row[grp_name] = sub_num[valid].median().median()
            class_group_median[cg] = row

        hm_df = pd.DataFrame(class_group_median).T.fillna(0)
        # Normalize each column to 0–1
        hm_norm = (hm_df - hm_df.min()) / (hm_df.max() - hm_df.min() + 1e-9)

        fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
        sns.heatmap(hm_norm, annot=hm_df.applymap(lambda x: f"{x:.1f}"),
                    fmt="", cmap="YlOrRd", linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Normalised Median"})
        ax.set_title("Feature Group Medians per Attack Class (row-normalised)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature Group")
        ax.set_ylabel("Attack Class Group")
        ax.tick_params(axis="x", labelsize=9, rotation=30)
        ax.tick_params(axis="y", labelsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / "feature_groups_summary.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ feature_groups_summary.png")

    print(f"\n✓ Done. Outputs → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="03 — Feature Analysis")
    parser.add_argument("--data-dir",   default="../nids_train/nids_dataset")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--sample",     type=float, default=0.10,
                        help="Fraction of data to sample (default 0.10 = 10%%)")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir), args.sample)
