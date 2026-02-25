"""
06_attack_profiles.py — Per-Attack Feature Fingerprinting
==========================================================
Understand what network-flow characteristics make each attack type unique
compared to benign traffic.

Outputs → nids_eda/outputs/
  attack_radar_charts.png         (spider/radar per attack vs BENIGN)
  attack_feature_heatmap.png      (attack × feature median normalised)
  attack_separation_ranking.csv   (features ranked by separation power per attack)
  top_separating_features.png     (bar chart of top features per attack class)

Usage:
    python 06_attack_profiles.py [--data-dir ..] [--output-dir ./outputs] [--sample 0.08]
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
             "Destination Port", "Protocol", "Timestamp"}

# Key features for radar chart (interpretable, important)
RADAR_FEATURES = [
    "Flow Duration",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd IAT Mean",
    "Packet Length Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

ATTACK_COLORS = [
    "#C44E52", "#E06C4E", "#DD8452", "#937860",
    "#8172B2", "#CCB974", "#64B5CD", "#4C72B0",
    "#55A868", "#C44E52", "#8C8C8C", "#6D904F",
    "#E377C2", "#7F7F7F",
]


def load_sample(data_dir: Path, frac: float) -> pd.DataFrame:
    frames = []
    for fname in DATASET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].str.strip()
        frames.append(df.sample(frac=frac, random_state=42))
    combined = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in combined.columns if c not in DROP_COLS]
    combined = combined[feat_cols].replace([np.inf, -np.inf], np.nan)
    return combined.dropna(subset=[c for c in combined.columns if c != "Label"])


def radar_chart(ax, values, categories, color, label, fill_alpha=0.15):
    """Draw a single radar trace on ax (must be polar)."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_c = values + [values[0]]
    angles_c  = angles + [angles[0]]
    ax.plot(angles_c, values_c, color=color, lw=2, label=label)
    ax.fill(angles_c, values_c, color=color, alpha=fill_alpha)
    ax.set_xticks(angles)
    ax.set_xticklabels([c[:15] for c in categories], size=6)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)


def main(data_dir: Path, out_dir: Path, frac: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("06 · ATTACK PROFILES")
    print("=" * 70)
    print(f"  Sample fraction: {frac:.1%}")

    df = load_sample(data_dir, frac)
    feat_cols = [c for c in df.columns if c != "Label"]
    all_labels = sorted(df["Label"].unique())
    attack_labels = [l for l in all_labels if l != "BENIGN"]
    print(f"  Loaded {len(df):,} rows | {len(attack_labels)} attack classes")

    # ── 1. Compute per-class median (normalised 0-1) ───────────────────────
    print("\nComputing per-class medians …")
    # Clip to p1–p99 per feature to reduce outlier impact on normalisation
    df_num = df[feat_cols].copy()
    for col in feat_cols:
        lo, hi = df_num[col].quantile(0.01), df_num[col].quantile(0.99)
        df_num[col] = df_num[col].clip(lo, hi)

    # Global min/max for normalisation
    global_min = df_num.min()
    global_max = df_num.max()
    global_range = (global_max - global_min).replace(0, 1)

    class_medians_raw = {}
    for lbl in all_labels:
        sub = df_num[df["Label"] == lbl]
        class_medians_raw[lbl] = sub.median()

    class_medians_norm = {
        lbl: (class_medians_raw[lbl] - global_min) / global_range
        for lbl in all_labels
    }

    # ── 2. Radar charts ───────────────────────────────────────────────────
    radar_feats = [f for f in RADAR_FEATURES if f in df_num.columns]
    benign_vals  = class_medians_norm["BENIGN"][radar_feats].tolist()

    n_attacks = len(attack_labels)
    n_cols = 4
    n_rows = math.ceil(n_attacks / n_cols)
    fig_h  = n_rows * 4.5

    fig = plt.figure(figsize=(n_cols * 5.5, fig_h), facecolor=BG)
    fig.suptitle("Per-Attack Radar Charts vs BENIGN Baseline",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, attack in enumerate(attack_labels):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, polar=True)
        ax.set_facecolor(BG)
        atk_vals = class_medians_norm[attack][radar_feats].tolist()
        radar_chart(ax, benign_vals, radar_feats, color="#4C72B0", label="BENIGN", fill_alpha=0.1)
        radar_chart(ax, atk_vals,    radar_feats, color=ATTACK_COLORS[idx % len(ATTACK_COLORS)],
                    label=attack[:20], fill_alpha=0.2)
        ax.set_title(attack[:28], fontsize=8, pad=12, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=6)

    plt.tight_layout()
    fig.savefig(out_dir / "attack_radar_charts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ attack_radar_charts.png")

    # ── 3. Attack × feature heatmap ───────────────────────────────────────
    # Use top-30 features by global variance for a readable heatmap
    top_feats = df_num.var().sort_values(ascending=False).head(30).index.tolist()

    hm_data = pd.DataFrame(
        {lbl: class_medians_norm[lbl][top_feats] for lbl in all_labels}
    ).T

    fig, ax = plt.subplots(figsize=(22, max(8, len(all_labels) * 0.7 + 2)), facecolor=BG)
    sns.heatmap(hm_data, cmap="YlOrRd", linewidths=0.3, ax=ax,
                cbar_kws={"label": "Normalised Median", "shrink": 0.6},
                xticklabels=True, yticklabels=True)
    ax.set_title("Attack Class vs Top-30 Features — Normalised Median Values",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Attack Class", fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=60)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "attack_feature_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ attack_feature_heatmap.png")

    # ── 4. Separation ranking CSV ─────────────────────────────────────────
    benign_med = class_medians_raw["BENIGN"]
    sep_rows = []
    for attack in attack_labels:
        atk_med = class_medians_raw[attack]
        delta = (atk_med - benign_med).abs() / (benign_med.abs() + 1e-9)
        for feat in feat_cols:
            sep_rows.append({
                "attack":    attack,
                "feature":   feat,
                "abs_delta": delta[feat],
                "atk_median": atk_med[feat],
                "ben_median": benign_med[feat],
            })
    sep_df = pd.DataFrame(sep_rows).sort_values(["attack", "abs_delta"], ascending=[True, False])
    sep_df.to_csv(out_dir / "attack_separation_ranking.csv", index=False)
    print(f"  ✓ attack_separation_ranking.csv  ({len(sep_df):,} rows)")

    # ── 5. Top-5 separating features per attack — small multiples bar chart
    n_top = 5
    n_cols2 = min(4, n_attacks)
    n_rows2 = math.ceil(n_attacks / n_cols2)
    fig, axes = plt.subplots(n_rows2, n_cols2,
                             figsize=(n_cols2 * 5, n_rows2 * 3.5), facecolor=BG)
    if n_attacks == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).flatten()
    fig.suptitle(f"Top-{n_top} Features Separating Each Attack from BENIGN",
                 fontsize=13, fontweight="bold")

    for idx, attack in enumerate(attack_labels):
        ax = axes[idx]
        ax.set_facecolor(BG)
        top = sep_df[sep_df["attack"] == attack].head(n_top)
        color = ATTACK_COLORS[idx % len(ATTACK_COLORS)]
        bars = ax.barh(range(len(top)), top["abs_delta"].values,
                       color=color, edgecolor="white", alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([f[:30] for f in top["feature"]], fontsize=7.5)
        ax.set_xlabel("Relative Δ from BENIGN", fontsize=7)
        ax.set_title(attack[:30], fontsize=8.5, fontweight="bold")
        ax.invert_yaxis()

    for j in range(n_attacks, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / "top_separating_features.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ top_separating_features.png")

    # ── Console table ──────────────────────────────────────────────────────
    print("\n  Top-3 separating features per attack:")
    print(f"  {'Attack':<42} {'Feature':<38} {'Δ':>8}")
    print("  " + "─" * 90)
    for attack in attack_labels:
        top3 = sep_df[sep_df["attack"] == attack].head(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            prefix = f"  {attack:<42}" if i == 0 else f"  {'':42}"
            print(f"{prefix} {row.feature:<38} {row.abs_delta:>8.3f}")
        print()

    print(f"✓ Done. Outputs → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="06 — Attack Profiles")
    parser.add_argument("--data-dir",   default="../nids_train/nids_dataset")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--sample",     type=float, default=0.08)
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir), args.sample)
