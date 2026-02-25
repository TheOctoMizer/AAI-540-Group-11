#!/usr/bin/env bash
# run_all_eda.sh — Run all NIDS EDA scripts in sequence
# Usage: bash run_all_eda.sh [--data-dir <path>] [--output-dir <path>]
# ============================================================

set -euo pipefail

DATA_DIR="../nids_train/nids_dataset"
OUTPUT_DIR="./outputs"

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)   DATA_DIR="$2";   shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "======================================================"
echo "  NIDS EDA Pipeline"
echo "  Data dir   : $DATA_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "======================================================"
echo ""

run_script() {
  local name="$1"
  local script="$2"
  shift 2
  local extra_args=("$@")

  echo "──────────────────────────────"
  echo "▶  $name"
  echo "──────────────────────────────"
  local start=$SECONDS
  python3 "$script" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" "${extra_args[@]+"${extra_args[@]}"}"
  echo "   Finished in $((SECONDS - start))s"
  echo ""
}

# Run each script
run_script "01 · Data Overview"        01_data_overview.py
run_script "02 · Class Distribution"   02_class_distribution.py
run_script "03 · Feature Analysis"     03_feature_analysis.py     --sample 0.10
run_script "04 · Correlation Analysis" 04_correlation_analysis.py --sample 0.05 --threshold 0.90
run_script "05 · Temporal Analysis"    05_temporal_analysis.py
run_script "06 · Attack Profiles"      06_attack_profiles.py      --sample 0.08
run_script "07 · Feature Importance"   07_feature_importance.py   --sample 0.05 --n-estimators 100

echo "======================================================"
echo "  ✓ ALL EDA SCRIPTS COMPLETE"
echo "  Outputs saved to: $OUTPUT_DIR"
echo "======================================================"
echo ""
echo "Key output files:"
for f in \
  data_summary_table.csv \
  class_imbalance_table.csv \
  high_correlation_pairs.csv \
  attack_separation_ranking.csv \
  top_features_ranking.csv; do
  if [[ -f "$OUTPUT_DIR/$f" ]]; then
    echo "  ✓ $OUTPUT_DIR/$f"
  fi
done
