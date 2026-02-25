"""
02_class_distribution.py — Class Imbalance Analysis
=====================================================
The single biggest challenge for NIDS ML systems is severe class imbalance.
This script makes it fully quantified and visual.

Outputs → nids_eda/outputs/
  class_distribution_bar.png        (linear + log scale side-by-side)
  per_day_class_heatmap.png          (attack type × day)
  temporal_stacked_proportion.png    (100% stacked bar per day)
  attack_category_pie.png            (grouped by attack family)
  class_imbalance_table.csv          (ratio stats)

Usage:
    python 02_class_distribution.py [--data-dir ..] [--output-dir ./outputs]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Palette & Style ──────────────────────────────────────────────────────────
BG = "#F8F9FA"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG})

# Attack family groupings
ATTACK_FAMILIES = {
    "BENIGN":                    "Benign",
    "DoS Hulk":                  "DoS",
    "DoS GoldenEye":             "DoS",
    "DoS slowloris":             "DoS",
    "DoS Slowhttptest":          "DoS",
    "Heartbleed":                "DoS",
    "DDoS":                      "DDoS",
    "PortScan":                  "Probe / Scan",
    "FTP-Patator":               "Brute Force",
    "SSH-Patator":               "Brute Force",
    "Bot":                       "Botnet",
    "Web Attack \xef\xbf\xbd Brute Force":  "Web Attack",
    "Web Attack \xef\xbf\xbd XSS":          "Web Attack",
    "Web Attack \xef\xbf\xbd Sql Injection": "Web Attack",
    "Infiltration":              "Infiltration",
}
# Fallback mapping for garbled encoding variants
def get_family(label: str) -> str:
    for key, fam in ATTACK_FAMILIES.items():
        if key.lower() in label.lower():
            return fam
    if "web attack" in label.lower():
        return "Web Attack"
    return "Other"

FAMILY_COLORS = {
    "Benign":       "#4C72B0",
    "DoS":          "#C44E52",
    "DDoS":         "#E06C4E",
    "Probe / Scan": "#DD8452",
    "Brute Force":  "#937860",
    "Botnet":       "#8172B2",
    "Web Attack":   "#CCB974",
    "Infiltration": "#64B5CD",
    "Other":        "#AAAAAA",
}

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


def load_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=lambda c: c.strip() == "Label", low_memory=False)
    df.columns = df.columns.str.strip()
    return df["Label"].str.strip()


def main(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("02 · CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # ── 1. Load labels from all files ──────────────────────────────────────
    per_day: dict[str, pd.Series] = {}
    for day, fname in DATASET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  ⚠  {fname} not found — skipping")
            continue
        per_day[day] = load_labels(fpath)
        print(f"  {day:<12}  {len(per_day[day]):>10,} rows")

    all_labels = pd.concat(per_day.values(), ignore_index=True)
    total = len(all_labels)

    # ── 2. Global class counts ─────────────────────────────────────────────
    counts = all_labels.value_counts().sort_values(ascending=False)
    pcts   = (counts / total * 100).round(4)

    print(f"\nTotal samples : {total:,}")
    print(f"Attack classes: {len(counts)}\n")
    print(f"{'Label':<45} {'Count':>10}  {'%':>7}")
    print("─" * 65)
    for lbl, cnt in counts.items():
        print(f"{lbl:<45} {cnt:>10,}  {pcts[lbl]:>6.3f}%")

    # ── 3. Bar chart — linear + log ────────────────────────────────────────
    families   = [get_family(l) for l in counts.index]
    bar_colors = [FAMILY_COLORS.get(f, "#AAAAAA") for f in families]
    labels_short = [l[:30] for l in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG)
    fig.suptitle("CICIDS2017 — Class Distribution (All Files Combined)", fontsize=14, fontweight="bold")

    for ax, scale, title in zip(axes, ["linear", "log"], ["Linear Scale", "Log Scale"]):
        bars = ax.bar(range(len(counts)), counts.values, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Number of Samples")
        ax.set_title(title, fontsize=11)
        if scale == "log":
            ax.set_yscale("log")
        # Annotate top 5
        for i, (bar, val) in enumerate(zip(bars, counts.values)):
            if i < 5 or val < 100:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                        f"{val:,}" if val >= 1000 else str(val),
                        ha="center", fontsize=7, rotation=0)

    # Legend for families
    handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()
               if f in families]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8,
                   title="Attack Family")

    plt.tight_layout()
    fig.savefig(out_dir / "class_distribution_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n  ✓ class_distribution_bar.png")

    # ── 4. Per-day × class heatmap ─────────────────────────────────────────
    all_classes = sorted(counts.index.tolist())
    heat_data = pd.DataFrame(0, index=all_classes, columns=list(per_day.keys()))
    for day, series in per_day.items():
        day_counts = series.value_counts()
        for cls, cnt in day_counts.items():
            if cls in heat_data.index:
                heat_data.loc[cls, day] = cnt

    # Normalize per class (row-wise) so rare classes are visible
    heat_norm = heat_data.div(heat_data.max(axis=1).replace(0, 1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), facecolor=BG)
    fig.suptitle("CICIDS2017 — Attack Type × Day", fontsize=14, fontweight="bold")

    sns.heatmap(heat_data, annot=True, fmt=",", cmap="Blues",
                linewidths=0.5, ax=axes[0], cbar_kws={"label": "Sample Count"})
    axes[0].set_title("Absolute Counts", fontsize=11)
    axes[0].set_ylabel("Attack Type")
    axes[0].tick_params(axis="y", labelsize=8)

    sns.heatmap(heat_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[1], cbar_kws={"label": "Relative (row-norm)"})
    axes[1].set_title("Row-Normalised (shows attack presence)", fontsize=11)
    axes[1].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "per_day_class_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ per_day_class_heatmap.png")

    # ── 5. Stacked 100% proportion bar ────────────────────────────────────
    prop_data = heat_data.div(heat_data.sum(axis=0), axis=1) * 100
    class_colors = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    bottom = np.zeros(len(per_day))
    for i, cls in enumerate(all_classes):
        vals = prop_data.loc[cls].values
        ax.bar(range(len(per_day)), vals, bottom=bottom,
               label=cls[:30], color=class_colors[i], edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xticks(range(len(per_day)))
    ax.set_xticklabels(list(per_day.keys()), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Attack Composition per Day (100% Stacked)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7, ncol=1,
              title="Attack Type", title_fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "temporal_stacked_proportion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ temporal_stacked_proportion.png")

    # ── 6. Attack family pie chart ─────────────────────────────────────────
    family_counts: dict[str, int] = {}
    for lbl, cnt in counts.items():
        fam = get_family(lbl)
        family_counts[fam] = family_counts.get(fam, 0) + cnt

    fam_series = pd.Series(family_counts).sort_values(ascending=False)
    fam_colors = [FAMILY_COLORS.get(f, "#AAAAAA") for f in fam_series.index]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        fam_series.values,
        labels=None,
        colors=fam_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
        startangle=140,
        pctdistance=0.8,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax.legend(wedges, [f"{n}\n({fam_series[n]:,})" for n in fam_series.index],
              loc="lower right", bbox_to_anchor=(1.35, 0), fontsize=9)
    ax.set_title("CICIDS2017 — Attack Family Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "attack_category_pie.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ attack_category_pie.png")

    # ── 7. Imbalance table CSV ─────────────────────────────────────────────
    majority_count = counts.max()
    imb_rows = []
    for lbl, cnt in counts.items():
        imb_rows.append({
            "label":          lbl,
            "family":         get_family(lbl),
            "count":          cnt,
            "pct":            pcts[lbl],
            "imbalance_ratio": majority_count / max(cnt, 1),
        })
    imb_df = pd.DataFrame(imb_rows).sort_values("count", ascending=False)
    imb_df.to_csv(out_dir / "class_imbalance_table.csv", index=False)
    print("  ✓ class_imbalance_table.csv")

    print(f"\n✓ Done. Outputs → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="02 — Class Distribution Analysis")
    parser.add_argument("--data-dir",   default="../nids_train/nids_dataset")
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir))
