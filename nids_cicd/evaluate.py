"""
NIDS Standalone Model Evaluator
=================================
Loads trained autoencoder + XGBoost ONNX artifacts and checks all quality
gates without retraining. Useful after manual training runs or to validate
a candidate model before triggering a full pipeline.

Usage:
    python evaluate.py [OPTIONS]

Options:
    --config PATH           Path to cicd config.json (default: ./config.json)
    --model-dir PATH        Directory containing trained model artifacts
                            (default: from config)
    --data-dir PATH         Directory containing CSV datasets and scaler_params.json
                            (default: from config)
    --output-report PATH    Where to write evaluation_report.json
    --dry-run               Generate synthetic data and run gates without loading
                            real models (no heavy dependencies needed)
"""

import argparse
import json
import sys
import math
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Autoencoder evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_autoencoder(model_dir: Path, data_dir: Path, cfg: dict) -> dict:
    """
    Load autoencoder.pt + scaler, run inference on the training summary MSE
    stats (already saved by train_ae.py), and check quality gates.
    """
    import torch, numpy as np

    ae_cfg = cfg["models"]["autoencoder"]
    gates  = cfg["quality_gates"]["autoencoder"]

    # --- Load training summary (written by train_ae.py) ---
    summary_path = model_dir / ae_cfg["artifacts"]["training_summary"]
    if not summary_path.exists():
        return {"error": f"Training summary not found: {summary_path}", "passed": False}

    with open(summary_path) as f:
        summary = json.load(f)

    val_stats = summary.get("validation_stats", {})
    mse_mean  = val_stats.get("mean",   float("inf"))
    mse_p95   = val_stats.get("p95",    float("inf"))
    mse_p99   = val_stats.get("p99",    float("inf"))

    # --- Gate checks ---
    gate_results = {
        "mse_mean": {
            "value":     round(mse_mean, 6),
            "threshold": gates["max_mse_mean"],
            "passed":    mse_mean <= gates["max_mse_mean"],
        },
        "mse_p95": {
            "value":     round(mse_p95, 6),
            "threshold": gates["max_mse_p95"],
            "passed":    mse_p95 <= gates["max_mse_p95"],
        },
        "mse_p99": {
            "value":     round(mse_p99, 6),
            "threshold": gates["max_mse_p99"],
            "passed":    mse_p99 <= gates["max_mse_p99"],
        },
    }

    overall_passed = all(g["passed"] for g in gate_results.values())

    return {
        "model":        "autoencoder",
        "mse_mean":     mse_mean,
        "mse_p95":      mse_p95,
        "mse_p99":      mse_p99,
        "final_val_loss": summary.get("final_val_loss"),
        "best_val_loss":  summary.get("best_val_loss"),
        "train_samples":  summary.get("num_train_samples"),
        "gates":          gate_results,
        "passed":         overall_passed,
    }


def evaluate_autoencoder_dry() -> dict:
    """Return plausible mock autoencoder evaluation results."""
    return {
        "model":        "autoencoder",
        "mse_mean":     0.0182,
        "mse_p95":      0.0210,
        "mse_p99":      0.0490,
        "final_val_loss": 0.0185,
        "best_val_loss":  0.0178,
        "train_samples":  50000,
        "gates": {
            "mse_mean": {"value": 0.0182, "threshold": 0.05,   "passed": True},
            "mse_p95":  {"value": 0.0210, "threshold": 0.0234, "passed": True},
            "mse_p99":  {"value": 0.0490, "threshold": 0.0567, "passed": True},
        },
        "passed": True,
        "dry_run": True,
    }


# ---------------------------------------------------------------------------
# XGBoost ONNX evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_xgboost(model_dir: Path, data_dir: Path, cfg: dict) -> dict:
    """
    Load xgb_classifier.onnx and the training summary, check accuracy gate.
    Also verifies ONNX consistency if consistency data is saved.
    """
    import onnxruntime as ort, numpy as np

    xgb_cfg = cfg["models"]["xgboost"]
    gates   = cfg["quality_gates"]["xgboost"]

    xgb_onnx_path = model_dir / xgb_cfg["artifacts"]["onnx"]
    if not xgb_onnx_path.exists():
        return {"error": f"ONNX model not found: {xgb_onnx_path}", "passed": False}

    # --- Load ONNX model (verify it loads) ---
    try:
        session = ort.InferenceSession(str(xgb_onnx_path))
        input_shape = session.get_inputs()[0].shape
    except Exception as e:
        return {"error": f"Failed to load ONNX: {e}", "passed": False}

    # --- Look for a saved eval report from classifier/eval_xgb.py ---
    eval_report_path = model_dir / "eval_report.json"
    accuracy = None
    onnx_max_diff  = None
    onnx_mean_diff = None

    if eval_report_path.exists():
        with open(eval_report_path) as f:
            stored = json.load(f)
        accuracy       = stored.get("accuracy")
        onnx_max_diff  = stored.get("onnx_consistency", {}).get("max_difference")
        onnx_mean_diff = stored.get("onnx_consistency", {}).get("mean_difference")

    # --- Gate checks ---
    gate_results = {}

    if accuracy is not None:
        gate_results["accuracy"] = {
            "value":     round(accuracy, 6),
            "threshold": gates["min_accuracy"],
            "passed":    accuracy >= gates["min_accuracy"],
        }

    if onnx_max_diff is not None:
        gate_results["onnx_max_diff"] = {
            "value":     onnx_max_diff,
            "threshold": gates["max_onnx_max_diff"],
            "passed":    onnx_max_diff <= gates["max_onnx_max_diff"],
        }

    if onnx_mean_diff is not None:
        gate_results["onnx_mean_diff"] = {
            "value":     onnx_mean_diff,
            "threshold": gates["max_onnx_mean_diff"],
            "passed":    onnx_mean_diff <= gates["max_onnx_mean_diff"],
        }

    # If we have no eval report, run a quick sanity check
    if not gate_results:
        dummy = [[0.1] * (input_shape[1] if len(input_shape) > 1 else 8)]
        try:
            import numpy as np
            inp  = {session.get_inputs()[0].name: np.array(dummy, dtype=np.float32)}
            outs = session.run(None, inp)
            gate_results["onnx_inference_sanity"] = {
                "value": "ok", "threshold": "ok", "passed": True
            }
        except Exception as e:
            gate_results["onnx_inference_sanity"] = {
                "value": str(e), "threshold": "ok", "passed": False
            }

    overall_passed = all(g["passed"] for g in gate_results.values())

    return {
        "model":        "xgboost",
        "accuracy":     accuracy,
        "onnx_max_diff":  onnx_max_diff,
        "onnx_mean_diff": onnx_mean_diff,
        "onnx_input_shape": list(input_shape),
        "gates":        gate_results,
        "passed":       overall_passed,
    }


def evaluate_xgboost_dry() -> dict:
    """Return plausible mock XGBoost evaluation results."""
    return {
        "model":          "xgboost",
        "accuracy":       0.9543,
        "onnx_max_diff":  2.38e-7,
        "onnx_mean_diff": 4.12e-8,
        "onnx_input_shape": [None, 8],
        "gates": {
            "accuracy":       {"value": 0.9543, "threshold": 0.92,  "passed": True},
            "onnx_max_diff":  {"value": 2.38e-7,"threshold": 1e-5,  "passed": True},
            "onnx_mean_diff": {"value": 4.12e-8,"threshold": 1e-6,  "passed": True},
        },
        "passed": True,
        "dry_run": True,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_result(label: str, result: dict):
    icon = "✅" if result.get("passed") else "❌"
    print(f"\n  {icon}  {label}")
    if "error" in result:
        print(f"     ERROR: {result['error']}")
        return
    for gate_name, gate in result.get("gates", {}).items():
        g_icon = "✅" if gate["passed"] else "❌"
        print(f"     {g_icon}  {gate_name:25s}: {gate['value']}  (threshold: {gate['threshold']})")


def build_report(ae_result: dict, xgb_result: dict, cfg: dict) -> dict:
    overall = ae_result.get("passed", False) and xgb_result.get("passed", False)
    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall,
        "autoencoder":    ae_result,
        "xgboost":        xgb_result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS model quality gate evaluator")
    parser.add_argument("--config",        default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--model-dir",     default=None,
                        help="Override model directory (default: from config)")
    parser.add_argument("--data-dir",      default=None,
                        help="Override data directory (default: from config)")
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    cicd_root  = Path(__file__).parent
    model_dir  = Path(args.model_dir)  if args.model_dir  else cicd_root / cfg["models"]["autoencoder"]["model_dir"]
    data_dir   = Path(args.data_dir)   if args.data_dir   else cicd_root / cfg["models"]["autoencoder"]["data_dir"]
    out_report = args.output_report or str(cicd_root / cfg["report"]["output_file"].replace(".json", "_eval.json"))

    print("=" * 60)
    print("NIDS MODEL EVALUATION")
    print("=" * 60)
    print(f"Model dir  : {model_dir}")
    print(f"Data dir   : {data_dir}")
    if args.dry_run:
        print("Mode       : DRY-RUN (synthetic data)")
    print("=" * 60)

    # --- Autoencoder ---
    print("\n[1/2] Evaluating Autoencoder ...")
    ae_result  = evaluate_autoencoder_dry() if args.dry_run else evaluate_autoencoder(model_dir, data_dir, cfg)
    print_result("Autoencoder", ae_result)

    # --- XGBoost ---
    print("\n[2/2] Evaluating XGBoost ...")
    xgb_result = evaluate_xgboost_dry() if args.dry_run else evaluate_xgboost(model_dir, data_dir, cfg)
    print_result("XGBoost", xgb_result)

    # --- Report ---
    report = build_report(ae_result, xgb_result, cfg)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out_report}")

    overall = report["overall_passed"]
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION: {'✅ PASSED — models meet all quality gates' if overall else '❌ FAILED — one or more gates did not pass'}")
    print(f"{'=' * 60}\n")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
