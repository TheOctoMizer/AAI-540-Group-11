"""
05_temporal_analysis.py — Per-Day Attack Patterns
==================================================
Understand how attacks are distributed across the days of the capture week
(Monday–Friday of the CICIDS2017 dataset).

Outputs → nids_eda/outputs/
  temporal_daily_counts.png      (row count + benign/attack split per day)
  temporal_attack_heatmap.png    (attack type × day grid)
  dataset_sizes.png              (file MB + row count comparison)

Usage:
    python 05_temporal_analysis.py [--data-dir ..] [--output-dir ./outputs]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

BG = "#F8F9FA"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG})

DATASET_FILES = [
    ("Monday",     "Monday-WorkingHours.pcap_ISCX.csv"),
    ("Tuesday",    "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("Wednesday",  "Wednesday-workingHours.pcap_ISCX.csv"),
    ("Thu-AM",     "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("Thu-PM",     "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("Fri-AM",     "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("Fri-DDoS",   "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    ("Fri-Scan",   "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
]

# Map day labels to logical weekday for grouping
DAY_MAP = {
    "Monday":    "Mon",
    "Tuesday":   "Tue",
    "Wednesday": "Wed",
    "Thu-AM":    "Thu",
    "Thu-PM":    "Thu",
    "Fri-AM":    "Fri",
    "Fri-DDoS":  "Fri",
    "Fri-Scan":  "Fri",
}


def load_labels_and_size(data_dir: Path):
    """Load only the Label column + file metadata from each CSV."""
    results = []
    for day, fname in DATASET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  ⚠  {fname} not found — skipping")
            continue
        mb = fpath.stat().st_size / 1_048_576
        df = pd.read_csv(fpath, usecols=lambda c: c.strip() == "Label", low_memory=False)
        df.columns = df.columns.str.strip()
        labels = df["Label"].str.strip()
        results.append({
            "day":      day,
            "fname":    fname,
            "mb":       mb,
            "n_rows":   len(labels),
            "n_benign": (labels == "BENIGN").sum(),
            "n_attack": (labels != "BENIGN").sum(),
            "labels":   labels,
        })
    return results


def main(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("05 · TEMPORAL / DAILY ANALYSIS")
    print("=" * 70)

    data = load_labels_and_size(data_dir)
    days  = [d["day"]    for d in data]
    rows  = [d["n_rows"] for d in data]
    mbs   = [d["mb"]     for d in data]
    benign = [d["n_benign"] for d in data]
    attack = [d["n_attack"] for d in data]

    # ── 1. Daily row counts stacked (benign vs attack) ─────────────────────
    x = np.arange(len(days))
    width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle("CICIDS2017 — Traffic Volume per Dataset File", fontsize=13, fontweight="bold")

    # Stacked bar: benign + attack
    axes[0].set_facecolor(BG)
    bars_b = axes[0].bar(x, benign, width, label="BENIGN",  color="#4C72B0", edgecolor="white")
    bars_a = axes[0].bar(x, attack, width, label="Attack",  color="#C44E52", edgecolor="white",
                         bottom=benign)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(days, rotation=40, ha="right", fontsize=9)
    axes[0].set_ylabel("Number of Flow Records")
    axes[0].set_title("Benign vs Attack Flows per File")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    for bar, val in zip(bars_b, rows):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + max(rows) * 0.01,
                     f"{val/1e3:.0f}k", ha="center", fontsize=7.5, fontweight="bold")

    # Attack % per file
    attack_pct = [a / r * 100 if r > 0 else 0 for a, r in zip(attack, rows)]
    axes[1].set_facecolor(BG)
    bar_pct = axes[1].bar(x, attack_pct, width, color="#DD8452", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(days, rotation=40, ha="right", fontsize=9)
    axes[1].set_ylabel("Attack Flow Percentage (%)")
    axes[1].set_title("% Attack Traffic per File")
    axes[1].set_ylim(0, max(attack_pct) * 1.25 if max(attack_pct) > 0 else 1)
    for bar, pct in zip(bar_pct, attack_pct):
        axes[1].text(bar.get_x() + bar.get_width() / 2, pct + 0.3,
                     f"{pct:.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "temporal_daily_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ temporal_daily_counts.png")

    # ── 2. File size comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)
    fig.suptitle("CICIDS2017 — Dataset File Sizes", fontsize=13, fontweight="bold")

    for ax, vals, lbl, color in zip(axes,
                                    [mbs, rows],
                                    ["Size (MB)", "Row Count"],
                                    ["#4C72B0", "#8172B2"]):
        ax.set_facecolor(BG)
        bars = ax.barh(days, vals, color=color, edgecolor="white")
        ax.set_xlabel(lbl)
        ax.set_title(f"Per-File {lbl}")
        for bar, val in zip(bars, vals):
            label_text = f"{val:.1f}" if isinstance(val, float) else f"{val:,}"
            ax.text(bar.get_width() + max(vals) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    label_text, va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "dataset_sizes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ dataset_sizes.png")

    # ── 3. Attack type × day heatmap ──────────────────────────────────────
    # Collect per-day, per-class counts
    all_labels_concat = pd.concat([d["labels"] for d in data], ignore_index=True)
    all_classes = sorted(all_labels_concat.unique().tolist())

    heat_abs  = pd.DataFrame(0, index=all_classes, columns=days, dtype=np.int64)
    for d in data:
        day_vc = d["labels"].value_counts()
        for cls, cnt in day_vc.items():
            if cls in heat_abs.index:
                heat_abs.loc[cls, d["day"]] = cnt

    # Row-normalise so rare classes are visible
    heat_norm = heat_abs.div(heat_abs.max(axis=1).replace(0, 1), axis=0)

    # Sort rows by total count desc  
    row_order = heat_abs.sum(axis=1).sort_values(ascending=False).index.tolist()
    heat_abs  = heat_abs.loc[row_order]
    heat_norm = heat_norm.loc[row_order]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=BG)
    fig.suptitle("CICIDS2017 — Attack Distribution Across Days", fontsize=13,
                 fontweight="bold")

    sns.heatmap(heat_abs, annot=True, fmt=",", cmap="Blues", linewidths=0.5,
                ax=axes[0], cbar_kws={"label": "Flow Count"},
                xticklabels=True, yticklabels=True)
    axes[0].set_title("Absolute Counts", fontsize=11)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].tick_params(axis="x", labelsize=8, rotation=35)

    sns.heatmap(heat_norm, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5,
                ax=axes[1], cbar_kws={"label": "Row-Normalised"},
                xticklabels=True, yticklabels=True)
    axes[1].set_title("Row-Normalised (highlights temporal presence)", fontsize=11)
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].tick_params(axis="x", labelsize=8, rotation=35)

    plt.tight_layout()
    fig.savefig(out_dir / "temporal_attack_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ temporal_attack_heatmap.png")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n  Per-file summary:")
    print(f"  {'Day':<14} {'Rows':>10} {'MB':>7} {'Benign':>10} {'Attack':>10} {'Atk%':>7}")
    print("  " + "─" * 62)
    for d, r, m, b, a, p in zip(days, rows, mbs, benign, attack, attack_pct):
        print(f"  {d:<14} {r:>10,} {m:>7.1f} {b:>10,} {a:>10,} {p:>6.1f}%")

    print(f"\n✓ Done. Outputs → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="05 — Temporal Analysis")
    parser.add_argument("--data-dir",   default="../nids_train/nids_dataset")
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir))
