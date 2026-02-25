"""
07_feature_importance.py — ML-Based Feature Ranking
=====================================================
Rank all 79 features using multiple importance metrics on a stratified sample:
  1. Mutual Information (sklearn) — measures non-linear statistical dependence
  2. Random Forest Gini Importance — captures interaction effects
  3. Combined rank — geometric mean of normalised MI + RF scores
  4. Variance (bonus) — trivially computable baseline

Outputs → nids_eda/outputs/
  feature_importance_mi.png           (MI bar chart, top 30)
  feature_importance_rf.png           (RF importance bar chart, top 30)
  feature_importance_combined.png     (side-by-side MI vs RF, normalised)
  feature_importance_variance.png     (variance-based rank, all features)
  top_features_ranking.csv            (combined table for all features)

Usage:
    python 07_feature_importance.py [--data-dir ..] [--output-dir ./outputs] [--sample 0.05] [--n-estimators 100]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BG = "#F8F9FA"
sns.set_theme(style="whitegrid", font_scale=1.05)
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


def load_stratified_sample(data_dir: Path, frac: float) -> pd.DataFrame:
    """Load and stratify-sample so rare classes aren't completely absent."""
    frames = []
    for fname in DATASET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].str.strip()
        # Oversample minority classes a bit within each file
        frames.append(df.groupby("Label", group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=42) if len(g) * frac >= 1 else g
        ))
    combined = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in combined.columns if c not in DROP_COLS]
    combined = combined[feat_cols].replace([np.inf, -np.inf], np.nan)
    # Drop rows with any NaN
    combined = combined.dropna()
    return combined


def plot_importance(scores: pd.Series, title: str, path: Path,
                    top_n: int = 30, color: str = "#4C72B0"):
    top = scores.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.32)), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(range(len(top)), top.values, color=color, edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Score (normalised)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    # Value labels
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + max(top.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(data_dir: Path, out_dir: Path, frac: float, n_estimators: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("07 · FEATURE IMPORTANCE")
    print("=" * 70)
    print(f"  Sample fraction: {frac:.1%}")
    print(f"  RF estimators  : {n_estimators}")

    # ── Load ───────────────────────────────────────────────────────────────
    df = load_stratified_sample(data_dir, frac)
    feat_cols = [c for c in df.columns if c != "Label"]
    X = df[feat_cols].values.astype(np.float32)
    y_raw = df["Label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"  Loaded {len(df):,} rows × {len(feat_cols)} features")
    print(f"  Classes: {len(le.classes_)}")

    # ── 1. Variance ────────────────────────────────────────────────────────
    print("\n[1/3] Computing feature variance …")
    var_scores = pd.Series(df[feat_cols].var().values, index=feat_cols)
    var_norm   = (var_scores - var_scores.min()) / (var_scores.max() - var_scores.min() + 1e-9)

    plot_importance(var_norm,
                    "Feature Importance — Variance (normalised, top-30)",
                    out_dir / "feature_importance_variance.png",
                    color="#937860")
    print("  ✓ feature_importance_variance.png")

    # ── 2. Mutual Information ──────────────────────────────────────────────
    print("[2/3] Computing Mutual Information (this may take ~60s) …")
    mi_scores_raw = mutual_info_classif(X, y, discrete_features=False,
                                        n_neighbors=5, random_state=42)
    mi_scores = pd.Series(mi_scores_raw, index=feat_cols)
    mi_norm   = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-9)

    plot_importance(mi_norm,
                    "Feature Importance — Mutual Information (normalised, top-30)",
                    out_dir / "feature_importance_mi.png",
                    color="#4C72B0")
    print("  ✓ feature_importance_mi.png")

    # ── 3. Random Forest ──────────────────────────────────────────────────
    print(f"[3/3] Training Random Forest ({n_estimators} trees, this may take 1–3 min) …")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X, y)
    rf_scores_raw = rf.feature_importances_
    rf_scores = pd.Series(rf_scores_raw, index=feat_cols)
    rf_norm   = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-9)

    plot_importance(rf_norm,
                    "Feature Importance — Random Forest Gini (normalised, top-30)",
                    out_dir / "feature_importance_rf.png",
                    color="#C44E52")
    print("  ✓ feature_importance_rf.png")

    # ── 4. Combined ranking (geometric mean) ──────────────────────────────
    combined_norm = np.sqrt(mi_norm * rf_norm)  # geometric mean
    combined_series = pd.Series(combined_norm, index=feat_cols)
    combined_sorted = combined_series.sort_values(ascending=False)

    # Side-by-side MI vs RF comparison chart (top 30 by combined rank)
    top30 = combined_sorted.head(30).index.tolist()
    mi_top  = mi_norm[top30]
    rf_top  = rf_norm[top30]

    x = np.arange(len(top30))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, max(8, len(top30) * 0.38)), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(x + width / 2, mi_top.values,  width, label="Mutual Info",    color="#4C72B0", alpha=0.85)
    ax.barh(x - width / 2, rf_top.values,  width, label="Random Forest",  color="#C44E52", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels([f[:38] for f in top30], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised Importance Score", fontsize=10)
    ax.set_title("Feature Importance — MI vs Random Forest (Top-30 by Combined Rank)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.axvline(0, color="black", lw=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance_combined.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ feature_importance_combined.png")

    # ── 5. Combined table CSV ─────────────────────────────────────────────
    ranking_df = pd.DataFrame({
        "feature":         feat_cols,
        "variance":        var_scores.values,
        "variance_norm":   var_norm.values,
        "mi_score":        mi_scores.values,
        "mi_norm":         mi_norm.values,
        "rf_importance":   rf_scores.values,
        "rf_norm":         rf_norm.values,
        "combined_score":  combined_norm,
        "combined_rank":   combined_series.rank(ascending=False).astype(int).values,
    })
    ranking_df = ranking_df.sort_values("combined_rank")
    ranking_df.to_csv(out_dir / "top_features_ranking.csv", index=False)
    print(f"  ✓ top_features_ranking.csv")

    # ── Console top-20 ────────────────────────────────────────────────────
    print("\n  TOP-20 FEATURES (combined MI + RF rank):")
    print(f"  {'Rank':>4}  {'Feature':<42} {'MI':>6}  {'RF':>6}  {'Combined':>8}")
    print("  " + "─" * 70)
    for _, row in ranking_df.head(20).iterrows():
        print(f"  {int(row.combined_rank):>4}  {row.feature:<42} "
              f"{row.mi_norm:>6.4f}  {row.rf_norm:>6.4f}  {row.combined_score:>8.4f}")

    # ── OOB accuracy proxy ────────────────────────────────────────────────
    print(f"\n  RF Training Accuracy: {rf.score(X, y) * 100:.2f}%  "
          f"(training set — use for sanity check only)")

    print(f"\n✓ Done. Outputs → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="07 — Feature Importance")
    parser.add_argument("--data-dir",      default="../nids_train/nids_dataset")
    parser.add_argument("--output-dir",    default="./outputs")
    parser.add_argument("--sample",        type=float, default=0.05,
                        help="Fraction of data to sample (default 0.05 = 5%%)")
    parser.add_argument("--n-estimators",  type=int, default=100,
                        help="Number of RF trees (default 100)")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir), args.sample, args.n_estimators)
