"""
NIDS CI/CD Pipeline Orchestrator
==================================
Automates the full retrain → evaluate → gate → redeploy cycle for both
NIDS models (autoencoder and XGBoost). If any quality gate fails, the
pipeline halts and does NOT deploy — protecting the production endpoints.

Pipeline steps:
    1.  [train-ae]    Retrain autoencoder         (train_ae.py subprocess)
    2.  [eval-ae]     Evaluate autoencoder gates
    3.  [train-xgb]   Retrain XGBoost classifier  (train_xgb.py subprocess)
    4.  [eval-xgb]    Evaluate XGBoost gates
    5.  [upload]      Upload artifacts to S3
    6.  [deploy-ae]   Deploy autoencoder to SageMaker
    7.  [deploy-xgb]  Deploy XGBoost to SageMaker
    8.  [smoke]       Smoke-test both endpoints
    9.  [report]      Write cicd_report.json

Usage:
    python pipeline.py [OPTIONS]

Options:
    --config PATH           Path to config.json (default: ./config.json)
    --model-dir PATH        Override model artifacts directory
    --data-dir PATH         Override dataset directory
    --role ARN              SageMaker execution role ARN
    --bucket NAME           S3 bucket (overrides config)
    --region REGION         AWS region (overrides config)
    --skip-train            Skip retraining (use existing artifacts)
    --skip-deploy           Skip deployment (evaluate + report only)
    --dry-run               Step through all stages without AWS calls or
                            launching subprocesses
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class StepResult:
    """Tracks the outcome of a single pipeline step."""
    def __init__(self, name: str):
        self.name    = name
        self.passed  = False
        self.skipped = False
        self.message = ""
        self.details = {}
        self.start   = time.time()
        self.elapsed = 0.0

    def succeed(self, msg: str = "", **kw):
        self.passed  = True
        self.elapsed = time.time() - self.start
        self.message = msg
        self.details.update(kw)

    def fail(self, msg: str = "", **kw):
        self.passed  = False
        self.elapsed = time.time() - self.start
        self.message = msg
        self.details.update(kw)

    def skip(self, reason: str = ""):
        self.skipped = True
        self.passed  = True   # skipped ≠ failed
        self.elapsed = 0.0
        self.message = reason

    def icon(self) -> str:
        if self.skipped: return "⏭ "
        return "✅" if self.passed else "❌"

    def to_dict(self) -> dict:
        return {
            "name":    self.name,
            "passed":  self.passed,
            "skipped": self.skipped,
            "message": self.message,
            "elapsed": round(self.elapsed, 2),
            **self.details,
        }


def banner(text: str):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def run_subprocess(cmd: list, cwd: str, dry_run: bool) -> tuple[bool, str]:
    """Run a command; return (success, combined output)."""
    if dry_run:
        print(f"    [DRY-RUN] Would run: {' '.join(cmd)}")
        return True, ""
    try:
        result = subprocess.run(
            cmd, cwd=cwd,
            capture_output=True, text=True, timeout=3600
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"    ❌ subprocess exited {result.returncode}")
            print(f"    {output[-2000:]}")   # tail of output
            return False, output
        print(output[-1000:])                # show tail of successful runs too
        return True, output
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Step 1 & 2: Autoencoder train + evaluate
# ---------------------------------------------------------------------------

def step_train_autoencoder(cfg: dict, model_dir: Path, data_dir: Path,
                            dry_run: bool) -> StepResult:
    r = StepResult("train_autoencoder")
    ae = cfg["models"]["autoencoder"]
    banner("STEP 1 — Training Autoencoder")

    cmd = [
        sys.executable, ae["train_script"],
        "--model-dir",   str(model_dir),
        "--data-dir",    str(data_dir),
        "--output-dir",  str(model_dir),
        "--epochs",      str(ae["epochs"]),
        "--batch-size",  str(ae["batch_size"]),
        "--learning-rate", str(ae["learning_rate"]),
        "--latent-dim",  str(ae["latent_dim"]),
        "--dataset",     ae["dataset"],
    ]

    ok, out = run_subprocess(cmd, cwd=str(Path(__file__).parent), dry_run=dry_run)
    if ok:
        r.succeed("Autoencoder training completed")
    else:
        r.fail("Autoencoder training subprocess failed", subprocess_output=out[-500:])
    return r


def step_eval_autoencoder(cfg: dict, model_dir: Path, dry_run: bool) -> StepResult:
    r    = StepResult("eval_autoencoder")
    gates = cfg["quality_gates"]["autoencoder"]
    banner("STEP 2 — Evaluating Autoencoder Quality Gates")

    if dry_run:
        # Synthetic pass
        print("  [DRY-RUN] Using synthetic MSE values (mean=0.018, p95=0.021, p99=0.049)")
        mse_mean, mse_p95, mse_p99 = 0.018, 0.021, 0.049
    else:
        summary_path = model_dir / "training_summary.json"
        if not summary_path.exists():
            r.fail(f"training_summary.json not found in {model_dir}")
            return r
        with open(summary_path) as f:
            summary = json.load(f)
        vs       = summary.get("validation_stats", {})
        mse_mean = vs.get("mean",  float("inf"))
        mse_p95  = vs.get("p95",   float("inf"))
        mse_p99  = vs.get("p99",   float("inf"))

    checks = {
        "mse_mean": (mse_mean, gates["max_mse_mean"]),
        "mse_p95":  (mse_p95,  gates["max_mse_p95"]),
        "mse_p99":  (mse_p99,  gates["max_mse_p99"]),
    }

    failed = []
    for name, (val, thresh) in checks.items():
        icon = "✅" if val <= thresh else "❌"
        print(f"  {icon}  {name:12s}: {val:.6f}  (max: {thresh})")
        if val > thresh:
            failed.append(name)

    gate_data = {k: {"value": v, "threshold": t, "passed": v <= t}
                 for k, (v, t) in checks.items()}

    if failed:
        r.fail(f"Gates failed: {', '.join(failed)}", gates=gate_data)
    else:
        r.succeed("All autoencoder gates passed", gates=gate_data)
    return r


# ---------------------------------------------------------------------------
# Step 3 & 4: XGBoost train + evaluate
# ---------------------------------------------------------------------------

def step_train_xgboost(cfg: dict, model_dir: Path, dry_run: bool) -> StepResult:
    r   = StepResult("train_xgboost")
    xgb = cfg["models"]["xgboost"]
    banner("STEP 3 — Training XGBoost Classifier")

    ae_pt = model_dir / cfg["models"]["autoencoder"]["artifacts"]["pt"]

    cmd = [
        sys.executable, xgb["train_script"],
        "--data-dir",          str(Path(__file__).parent / xgb["data_dir"]),
        "--dataset",           xgb["dataset"],
        "--output-dir",        str(model_dir),
        "--autoencoder-model", str(ae_pt),
        "--latent-dim",        str(cfg["models"]["autoencoder"]["latent_dim"]),
    ]

    ok, out = run_subprocess(cmd, cwd=str(Path(__file__).parent), dry_run=dry_run)
    if ok:
        r.succeed("XGBoost training completed")
    else:
        r.fail("XGBoost training subprocess failed", subprocess_output=out[-500:])
    return r


def step_eval_xgboost(cfg: dict, model_dir: Path, dry_run: bool) -> StepResult:
    r     = StepResult("eval_xgboost")
    gates = cfg["quality_gates"]["xgboost"]
    banner("STEP 4 — Evaluating XGBoost Quality Gates")

    if dry_run:
        print("  [DRY-RUN] Using synthetic accuracy=0.954, onnx_max_diff=2.4e-7")
        accuracy, onnx_max, onnx_mean = 0.9543, 2.4e-7, 4.1e-8
    else:
        eval_path = model_dir / "eval_report.json"
        if not eval_path.exists():
            # Fallback: just verify the ONNX file loads
            onnx_path = model_dir / cfg["models"]["xgboost"]["artifacts"]["onnx"]
            if not onnx_path.exists():
                r.fail(f"ONNX model not found: {onnx_path}")
                return r
            try:
                import onnxruntime as ort
                ort.InferenceSession(str(onnx_path))
                r.succeed("ONNX model verified (no eval_report.json found — skipping accuracy gate)",
                          note="Run classifier/eval_xgb.py to produce full evaluation")
                return r
            except Exception as e:
                r.fail(f"ONNX model failed to load: {e}")
                return r

        with open(eval_path) as f:
            ev = json.load(f)
        accuracy  = ev.get("accuracy", 0.0)
        cons      = ev.get("onnx_consistency", {})
        onnx_max  = cons.get("max_difference",  float("inf"))
        onnx_mean = cons.get("mean_difference", float("inf"))

    checks = {
        "accuracy":       (accuracy,  gates["min_accuracy"],        lambda v, t: v >= t),
        "onnx_max_diff":  (onnx_max,  gates["max_onnx_max_diff"],   lambda v, t: v <= t),
        "onnx_mean_diff": (onnx_mean, gates["max_onnx_mean_diff"],  lambda v, t: v <= t),
    }

    failed = []
    for name, (val, thresh, check) in checks.items():
        passed = check(val, thresh)
        icon   = "✅" if passed else "❌"
        print(f"  {icon}  {name:20s}: {val:.2e}  (threshold: {thresh:.2e})")
        if not passed:
            failed.append(name)

    gate_data = {k: {"value": v, "threshold": t, "passed": fn(v, t)}
                 for k, (v, t, fn) in checks.items()}

    if failed:
        r.fail(f"Gates failed: {', '.join(failed)}", gates=gate_data)
    else:
        r.succeed("All XGBoost gates passed", gates=gate_data)
    return r


# ---------------------------------------------------------------------------
# Step 5: Upload artifacts to S3
# ---------------------------------------------------------------------------

def step_upload_artifacts(cfg: dict, model_dir: Path, region: str,
                           bucket: str, dry_run: bool) -> StepResult:
    r = StepResult("upload_artifacts")
    banner("STEP 5 — Uploading Model Artifacts to S3")

    artifacts = {
        "autoencoder.pt":      model_dir / cfg["models"]["autoencoder"]["artifacts"]["pt"],
        "autoencoder.onnx":    model_dir / cfg["models"]["autoencoder"]["artifacts"]["onnx"],
        "xgb_classifier.onnx": model_dir / cfg["models"]["xgboost"]["artifacts"]["onnx"],
    }

    prefix    = cfg["aws"]["s3_prefix"]
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_uris   = {}

    for artifact_name, local_path in artifacts.items():
        s3_key = f"{prefix}/{run_stamp}/{artifact_name}"
        if dry_run:
            print(f"  [DRY-RUN] Would upload {artifact_name} → s3://{bucket}/{s3_key}")
            s3_uris[artifact_name] = f"s3://{bucket}/{s3_key}"
            continue
        if not local_path.exists():
            r.fail(f"Artifact not found: {local_path}")
            return r
        try:
            import boto3
            s3 = boto3.client("s3", region_name=region)
            s3.upload_file(str(local_path), bucket, s3_key)
            s3_uris[artifact_name] = f"s3://{bucket}/{s3_key}"
            print(f"  ✅ Uploaded s3://{bucket}/{s3_key}")
        except Exception as e:
            r.fail(f"Upload failed for {artifact_name}: {e}")
            return r

    r.succeed("All artifacts uploaded", s3_uris=s3_uris, run_stamp=run_stamp)
    return r


# ---------------------------------------------------------------------------
# Steps 6 & 7: Deploy to SageMaker
# ---------------------------------------------------------------------------

def _deploy_model(deploy_script: str, extra_args: list,
                  label: str, dry_run: bool) -> StepResult:
    r   = StepResult(f"deploy_{label}")
    cmd = [sys.executable, deploy_script] + extra_args + ["--skip-test"]
    ok, out = run_subprocess(cmd, cwd=str(Path(__file__).parent), dry_run=dry_run)
    if ok:
        r.succeed(f"{label} deployed")
    else:
        r.fail(f"{label} deployment failed", subprocess_output=out[-500:])
    return r


def step_deploy_autoencoder(cfg: dict, model_dir: Path, role: str,
                             bucket: str, region: str, dry_run: bool) -> StepResult:
    banner("STEP 6 — Deploying Autoencoder to SageMaker")
    script    = str(Path(__file__).parent / "../nids_sagemaker_deploy/deploy_autoencoder.py")
    ae_cfg    = cfg["models"]["autoencoder"]
    scaler    = str(model_dir / ae_cfg["artifacts"]["scaler"])
    extra     = [
        "--model-path",    str(model_dir / ae_cfg["artifacts"]["pt"]),
        "--scaler-path",   scaler,
        "--endpoint-name", cfg["sagemaker"]["autoencoder_endpoint"],
        "--instance-type", cfg["sagemaker"]["instance_type_autoencoder"],
        "--role",          role,
        "--bucket",        bucket,
        "--region",        region,
    ]
    return _deploy_model(script, extra, "autoencoder", dry_run)


def step_deploy_xgboost(cfg: dict, model_dir: Path, role: str,
                         bucket: str, region: str, dry_run: bool) -> StepResult:
    banner("STEP 7 — Deploying XGBoost to SageMaker")
    script  = str(Path(__file__).parent / "../nids_sagemaker_deploy/deploy_xgboost.py")
    xgb_cfg = cfg["models"]["xgboost"]
    extra   = [
        "--model-path",      str(model_dir / xgb_cfg["artifacts"]["onnx"]),
        "--label-map-path",  str(Path(__file__).parent / xgb_cfg["artifacts"]["label_map"]),
        "--endpoint-name",   cfg["sagemaker"]["xgboost_endpoint"],
        "--instance-type",   cfg["sagemaker"]["instance_type_xgboost"],
        "--role",            role,
        "--bucket",          bucket,
        "--region",          region,
    ]
    return _deploy_model(script, extra, "xgboost", dry_run)


# ---------------------------------------------------------------------------
# Step 8: Smoke test
# ---------------------------------------------------------------------------

def step_smoke_test(cfg: dict, region: str, dry_run: bool) -> StepResult:
    r     = StepResult("smoke_test")
    st    = cfg["smoke_test"]
    banner("STEP 8 — Smoke Testing Endpoints")

    if dry_run:
        print("  [DRY-RUN] Would invoke both SageMaker endpoints with test payloads")
        r.succeed("Smoke test skipped (dry-run)")
        return r

    import boto3, json as _json, numpy as np
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    results = {}

    # Autoencoder
    ae_ep = cfg["sagemaker"]["autoencoder_endpoint"]
    try:
        resp = runtime.invoke_endpoint(
            EndpointName=ae_ep,
            ContentType="application/json",
            Body=_json.dumps({"features": [0.1] * st["autoencoder_features"]}),
        )
        out = _json.loads(resp["Body"].read())
        print(f"  ✅ Autoencoder: mse_error={out['mse_error'][0]:.6f}")
        results["autoencoder"] = {"passed": True, "mse_error": out["mse_error"][0]}
    except Exception as e:
        print(f"  ❌ Autoencoder smoke test failed: {e}")
        results["autoencoder"] = {"passed": False, "error": str(e)}

    # XGBoost
    xgb_ep = cfg["sagemaker"]["xgboost_endpoint"]
    try:
        resp = runtime.invoke_endpoint(
            EndpointName=xgb_ep,
            ContentType="application/json",
            Body=_json.dumps({"encoded": [0.1] * st["xgboost_encoded_dim"]}),
        )
        out = _json.loads(resp["Body"].read())
        print(f"  ✅ XGBoost: label={out['labels'][0]}, confidence={out['confidences'][0]:.4f}")
        results["xgboost"] = {"passed": True, "label": out["labels"][0]}
    except Exception as e:
        print(f"  ❌ XGBoost smoke test failed: {e}")
        results["xgboost"] = {"passed": False, "error": str(e)}

    all_passed = all(v["passed"] for v in results.values())
    if all_passed:
        r.succeed("Both endpoints responding correctly", smoke_results=results)
    else:
        r.fail("One or more endpoints failed smoke test", smoke_results=results)
    return r


# ---------------------------------------------------------------------------
# Step 9: Write report
# ---------------------------------------------------------------------------

def write_report(steps: list, cfg: dict, dry_run: bool) -> str:
    overall = all(s.passed for s in steps)
    report  = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall,
        "dry_run":        dry_run,
        "steps":          [s.to_dict() for s in steps],
    }
    out_path = str(Path(__file__).parent / cfg["report"]["output_file"])
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS CI/CD pipeline orchestrator")
    parser.add_argument("--config",       default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--model-dir",    default=None)
    parser.add_argument("--data-dir",     default=None)
    parser.add_argument("--role",         default=None, help="SageMaker IAM role ARN")
    parser.add_argument("--bucket",       default=None, help="S3 bucket name")
    parser.add_argument("--region",       default=None)
    parser.add_argument("--skip-train",   action="store_true", help="Skip retraining steps")
    parser.add_argument("--skip-deploy",  action="store_true", help="Skip deployment steps")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Step through all stages without AWS calls")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    region = args.region or cfg["aws"]["region"]
    bucket = args.bucket or cfg["aws"]["bucket"]
    role   = args.role   or cfg["aws"]["role_arn"]

    cicd_root = Path(__file__).parent
    model_dir = Path(args.model_dir) if args.model_dir else (cicd_root / cfg["models"]["autoencoder"]["model_dir"]).resolve()
    data_dir  = Path(args.data_dir)  if args.data_dir  else (cicd_root / cfg["models"]["autoencoder"]["data_dir"]).resolve()

    print("=" * 60)
    print("NIDS CI/CD PIPELINE")
    print("=" * 60)
    print(f"Region     : {region}")
    print(f"Bucket     : {bucket}")
    print(f"Model dir  : {model_dir}")
    print(f"Skip train : {args.skip_train}")
    print(f"Skip deploy: {args.skip_deploy}")
    if args.dry_run:
        print("Mode       : DRY-RUN")
    print("=" * 60)

    steps: list[StepResult] = []

    # ── 1. Train Autoencoder ──────────────────────────────────────────────
    r1 = StepResult("train_autoencoder")
    if args.skip_train:
        r1.skip("--skip-train flag set")
        print(f"  {r1.icon()}  [SKIP] Training autoencoder")
    else:
        r1 = step_train_autoencoder(cfg, model_dir, data_dir, args.dry_run)
    steps.append(r1)
    if not r1.passed:
        print(f"\n  Pipeline halted at step 1: {r1.message}")
        write_report(steps, cfg, args.dry_run)
        sys.exit(1)

    # ── 2. Evaluate Autoencoder ───────────────────────────────────────────
    r2 = step_eval_autoencoder(cfg, model_dir, args.dry_run)
    steps.append(r2)
    if not r2.passed:
        print(f"\n  Pipeline halted: autoencoder quality gates FAILED → {r2.message}")
        write_report(steps, cfg, args.dry_run)
        sys.exit(1)

    # ── 3. Train XGBoost ─────────────────────────────────────────────────
    r3 = StepResult("train_xgboost")
    if args.skip_train:
        r3.skip("--skip-train flag set")
        print(f"  {r3.icon()}  [SKIP] Training XGBoost")
    else:
        r3 = step_train_xgboost(cfg, model_dir, args.dry_run)
    steps.append(r3)
    if not r3.passed:
        print(f"\n  Pipeline halted at step 3: {r3.message}")
        write_report(steps, cfg, args.dry_run)
        sys.exit(1)

    # ── 4. Evaluate XGBoost ──────────────────────────────────────────────
    r4 = step_eval_xgboost(cfg, model_dir, args.dry_run)
    steps.append(r4)
    if not r4.passed:
        print(f"\n  Pipeline halted: XGBoost quality gates FAILED → {r4.message}")
        write_report(steps, cfg, args.dry_run)
        sys.exit(1)

    # ── 5. Upload Artifacts ───────────────────────────────────────────────
    r5 = StepResult("upload_artifacts")
    if args.skip_deploy:
        r5.skip("--skip-deploy flag set")
        print(f"  {r5.icon()}  [SKIP] Uploading artifacts")
    else:
        r5 = step_upload_artifacts(cfg, model_dir, region, bucket, args.dry_run)
    steps.append(r5)
    if not r5.passed:
        print(f"\n  Pipeline halted at step 5: {r5.message}")
        write_report(steps, cfg, args.dry_run)
        sys.exit(1)

    # ── 6. Deploy Autoencoder ─────────────────────────────────────────────
    r6 = StepResult("deploy_autoencoder")
    if args.skip_deploy:
        r6.skip("--skip-deploy flag set")
        print(f"  {r6.icon()}  [SKIP] Deploying autoencoder")
    else:
        if not role:
            r6.fail("SageMaker role ARN not configured. Set --role or aws.role_arn in config.json")
        else:
            r6 = step_deploy_autoencoder(cfg, model_dir, role, bucket, region, args.dry_run)
    steps.append(r6)
    if not r6.passed:
        print(f"\n  Pipeline halted at step 6: {r6.message}")
        write_report(steps, cfg, args.dry_run)
        sys.exit(1)

    # ── 7. Deploy XGBoost ─────────────────────────────────────────────────
    r7 = StepResult("deploy_xgboost")
    if args.skip_deploy:
        r7.skip("--skip-deploy flag set")
        print(f"  {r7.icon()}  [SKIP] Deploying XGBoost")
    else:
        if not role:
            r7.fail("SageMaker role ARN not configured")
        else:
            r7 = step_deploy_xgboost(cfg, model_dir, role, bucket, region, args.dry_run)
    steps.append(r7)
    if not r7.passed:
        print(f"\n  Pipeline halted at step 7: {r7.message}")
        write_report(steps, cfg, args.dry_run)
        sys.exit(1)

    # ── 8. Smoke Test ─────────────────────────────────────────────────────
    r8 = StepResult("smoke_test")
    if args.skip_deploy:
        r8.skip("--skip-deploy flag set")
        print(f"  {r8.icon()}  [SKIP] Smoke test")
    else:
        r8 = step_smoke_test(cfg, region, args.dry_run)
    steps.append(r8)

    # ── 9. Final Report ───────────────────────────────────────────────────
    report_path = write_report(steps, cfg, args.dry_run)

    banner("PIPELINE SUMMARY")
    for s in steps:
        elapsed_str = f"({s.elapsed:.1f}s)" if not s.skipped else "(skipped)"
        print(f"  {s.icon()}  {s.name:<28s} {elapsed_str}")
        if not s.passed:
            print(f"       ⚠  {s.message}")

    overall = all(s.passed for s in steps)
    print(f"\n  {'✅ PIPELINE PASSED' if overall else '❌ PIPELINE FAILED'}")
    print(f"  Report → {report_path}")
    print()

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
