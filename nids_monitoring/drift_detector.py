"""
NIDS MSE Drift Detector
========================
Detects statistical drift in autoencoder MSE reconstruction error over time.

Accepts a CSV file with a column of MSE values (produced by your Rust NIDS
client, test pipeline, or the monitor log). Uses a sliding window to compute
rolling mean + std and flags samples that are statistically anomalous.

Optionally emits a custom CloudWatch metric so you can graph drift in AWS.

Usage:
    python drift_detector.py [OPTIONS]

Options:
    --config PATH           Path to config.json (default: ./config.json)
    --input-csv PATH        CSV file with MSE values (required, or --from-logs)
    --from-logs PATH        Read MSE values from monitor.jsonl log (autoencoder
                            latency is used as a proxy when no CSV is available)
    --mse-column NAME       Column name in CSV that contains MSE values
                            (default: mse_error)
    --window INT            Rolling window size (overrides config)
    --output-report PATH    Write JSON drift report (overrides config)
    --emit-cloudwatch       Emit a custom CloudWatch metric for drift score
    --region REGION         AWS region for CloudWatch (overrides config)
    --dry-run               Compute drift but skip CloudWatch emission
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mse_from_csv(csv_path: str, column: str = "mse_error") -> list:
    """Load MSE values from a CSV file."""
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            # Try first numeric column as fallback
            headers = reader.fieldnames or []
            f.seek(0)
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                try:
                    values.append(float(row[0]))
                except (ValueError, IndexError):
                    pass
            print(f"  [WARN] Column '{column}' not found; using first column.")
            return values

        for row in reader:
            try:
                values.append(float(row[column]))
            except (ValueError, KeyError):
                pass
    return values


def load_mse_from_logs(log_path: str) -> list:
    """
    Extract a proxy latency series from a monitor JSONL log.
    Uses ModelLatency_Average of the autoencoder endpoint.
    """
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {log_path}")
        return []

    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
                ae = snap.get("endpoints", {}).get("autoencoder", {})
                lat = ae.get("metrics", {}).get("ModelLatency_Average")
                if isinstance(lat, (int, float)):
                    # Normalise µs → a unit comparable to MSE thresholds
                    values.append(lat / 1_000_000)
            except json.JSONDecodeError:
                pass
    return values


# ---------------------------------------------------------------------------
# Drift statistics
# ---------------------------------------------------------------------------

def moving_stats(values: list, window: int):
    """Compute rolling mean and std over a sliding window (simple Python, no pandas)."""
    import math
    means, stds = [], []
    for i in range(len(values)):
        w = values[max(0, i - window + 1): i + 1]
        mu = sum(w) / len(w)
        variance = sum((x - mu) ** 2 for x in w) / len(w)
        means.append(mu)
        stds.append(math.sqrt(variance))
    return means, stds


def detect_drift(values: list, window: int, z_threshold: float, thresholds: dict) -> dict:
    """
    Slide a window over MSE values and flag drift events.

    Returns a drift report dict.
    """
    import math

    if not values:
        return {"error": "No values to analyse."}

    means, stds = moving_stats(values, window)

    drift_events = []
    for i, v in enumerate(values):
        mu, sigma = means[i], stds[i]
        z = (v - mu) / sigma if sigma > 1e-9 else 0.0
        if abs(z) > z_threshold:
            drift_events.append({"index": i, "value": v, "z_score": round(z, 4)})

    overall_mean = sum(values) / len(values)
    overall_std  = math.sqrt(sum((x - overall_mean) ** 2 for x in values) / len(values))

    p95 = thresholds.get("p95", 0.0234)
    p99 = thresholds.get("p99", 0.0567)

    baseline_ok = overall_mean < p95
    severity = "OK"
    if overall_mean >= thresholds.get("p99_9", 0.0891):
        severity = "CRITICAL"
    elif overall_mean >= p99:
        severity = "HIGH"
    elif overall_mean >= p95:
        severity = "ELEVATED"

    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "samples":            len(values),
        "window_size":        window,
        "z_threshold":        z_threshold,
        "overall_mean":       round(overall_mean, 6),
        "overall_std":        round(overall_std, 6),
        "drift_event_count":  len(drift_events),
        "drift_rate_pct":     round(len(drift_events) / len(values) * 100, 2),
        "severity":           severity,
        "thresholds_used":    thresholds,
        "drift_events":       drift_events[:50],  # cap list in report
    }


# ---------------------------------------------------------------------------
# CloudWatch emission
# ---------------------------------------------------------------------------

def emit_cloudwatch_metric(report: dict, region: str, dry_run: bool):
    """Emit a custom CloudWatch metric for drift score."""
    drift_rate = report.get("drift_rate_pct", 0.0)
    severity   = report.get("severity", "OK")

    if dry_run:
        print(f"[DRY-RUN] Would emit CloudWatch metric:")
        print(f"  Namespace : NIDS/DriftDetection")
        print(f"  Metric    : DriftRatePct = {drift_rate}")
        print(f"  Severity  : {severity}")
        return

    try:
        import boto3
        cw = boto3.client("cloudwatch", region_name=region)
        cw.put_metric_data(
            Namespace="NIDS/DriftDetection",
            MetricData=[
                {
                    "MetricName": "DriftRatePct",
                    "Value":      drift_rate,
                    "Unit":       "Percent",
                    "Dimensions": [{"Name": "Model", "Value": "autoencoder"}],
                },
                {
                    "MetricName": "MeanMSE",
                    "Value":      report.get("overall_mean", 0.0),
                    "Unit":       "None",
                    "Dimensions": [{"Name": "Model", "Value": "autoencoder"}],
                },
            ],
        )
        print("  ✅ CloudWatch metrics emitted (namespace: NIDS/DriftDetection)")
    except Exception as e:
        print(f"  ❌ CloudWatch emission failed: {e}")


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(report: dict):
    sev_icons = {"OK": "✅", "ELEVATED": "⚠️", "HIGH": "🔶", "CRITICAL": "🚨"}
    icon = sev_icons.get(report.get("severity", "OK"), "❓")
    print(f"\n  Severity          : {icon}  {report.get('severity')}")
    print(f"  Samples analysed  : {report.get('samples')}")
    print(f"  Window size       : {report.get('window_size')}")
    print(f"  Overall mean MSE  : {report.get('overall_mean')}")
    print(f"  Overall std MSE   : {report.get('overall_std')}")
    print(f"  Drift events      : {report.get('drift_event_count')}  "
          f"({report.get('drift_rate_pct')}%)")

    # Show most anomalous events
    events = report.get("drift_events", [])
    if events:
        print(f"\n  Top drift events (by |z|):")
        top = sorted(events, key=lambda e: abs(e["z_score"]), reverse=True)[:5]
        for ev in top:
            print(f"    idx={ev['index']:6d}  mse={ev['value']:.6f}  z={ev['z_score']:+.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS MSE drift detector")
    parser.add_argument("--config",           default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--input-csv",        default=None, help="CSV file with MSE values")
    parser.add_argument("--from-logs",        default=None, help="monitor.jsonl log file")
    parser.add_argument("--mse-column",       default="mse_error")
    parser.add_argument("--window",           type=int, default=None)
    parser.add_argument("--output-report",    default=None)
    parser.add_argument("--emit-cloudwatch",  action="store_true")
    parser.add_argument("--region",           default=None)
    parser.add_argument("--dry-run",          action="store_true")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    region = args.region or cfg["aws"]["region"]
    window = args.window or cfg["drift"]["window_size"]
    z_thr  = cfg["drift"]["z_score_threshold"]
    out    = args.output_report or str(Path(__file__).parent / cfg["drift"]["output_report"])

    print("=" * 60)
    print("NIDS DRIFT DETECTOR")
    print("=" * 60)
    if args.dry_run:
        print("Mode   : DRY-RUN")

    # --- Load values ---
    if args.dry_run and not args.input_csv and not args.from_logs:
        import random
        print("Generating synthetic MSE values for dry-run ...")
        values = [random.gauss(0.02, 0.005) for _ in range(200)]
        # Inject some drift
        values += [random.gauss(0.08, 0.01) for _ in range(20)]
    elif args.input_csv:
        print(f"Loading MSE values from CSV: {args.input_csv}")
        values = load_mse_from_csv(args.input_csv, args.mse_column)
    elif args.from_logs:
        print(f"Loading values from monitor log: {args.from_logs}")
        values = load_mse_from_logs(args.from_logs)
    else:
        # Default to log file location from config
        log_file = Path(__file__).parent / cfg["monitor"]["log_dir"] / cfg["monitor"]["log_filename"]
        print(f"Loading values from default monitor log: {log_file}")
        values = load_mse_from_logs(str(log_file))

    if not values:
        print("No values loaded — nothing to analyse.")
        return

    print(f"Loaded {len(values)} values  |  window={window}  |  z_threshold={z_thr}")

    # --- Detect ---
    report = detect_drift(values, window, z_thr, cfg["thresholds"]["mse"])
    print_report(report)

    # --- Save report ---
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out}")

    # --- Optionally emit CloudWatch ---
    if args.emit_cloudwatch or args.dry_run:
        emit_cloudwatch_metric(report, region, dry_run=args.dry_run)

    # Exit with non-zero if drift is HIGH or CRITICAL
    severity = report.get("severity", "OK")
    if severity in ("HIGH", "CRITICAL"):
        sys.exit(1)


if __name__ == "__main__":
    main()
