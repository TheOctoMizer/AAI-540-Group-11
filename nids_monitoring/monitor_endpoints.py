"""
NIDS SageMaker Endpoint Monitor
================================
Polls CloudWatch metrics for both nids-autoencoder and nids-xgboost endpoints
and optionally writes results to a JSONL log file.

Usage:
    python monitor_endpoints.py [OPTIONS]

Options:
    --config PATH       Path to config.json (default: ./config.json)
    --region REGION     AWS region (overrides config)
    --interval SECS     Poll interval in seconds (0 = run once and exit)
    --log-file PATH     JSONL file to append results to
    --dry-run           Print what would be queried without calling AWS
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CloudWatch helpers
# ---------------------------------------------------------------------------

ENDPOINT_METRICS = [
    # (MetricName, Statistic, unit)
    ("Invocations",          "Sum",     "Count"),
    ("ModelLatency",         "Average", "Microseconds"),
    ("ModelLatency",         "p90",     "Microseconds"),
    ("ModelLatency",         "p99",     "Microseconds"),
    ("OverheadLatency",      "Average", "Microseconds"),
    ("Invocation4XXErrors",  "Sum",     "Count"),
    ("Invocation5XXErrors",  "Sum",     "Count"),
    ("InvocationsPerInstance","Sum",    "Count"),
]


def fetch_cloudwatch_metrics(
    cw_client,
    endpoint_name: str,
    variant_name: str,
    lookback_minutes: int,
) -> dict:
    """Pull the most recent data point for each endpoint metric from CloudWatch."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=lookback_minutes)

    results = {}
    for metric_name, stat, _ in ENDPOINT_METRICS:
        try:
            kwargs = {
                "Namespace": "AWS/SageMaker",
                "MetricName": metric_name,
                "Dimensions": [
                    {"Name": "EndpointName",  "Value": endpoint_name},
                    {"Name": "VariantName",   "Value": variant_name},
                ],
                "StartTime": start,
                "EndTime": now,
                "Period": lookback_minutes * 60,
            }
            if stat.startswith("p"):
                kwargs["ExtendedStatistics"] = [stat]
            else:
                kwargs["Statistics"] = [stat]

            resp = cw_client.get_metric_statistics(**kwargs)
            dps = resp.get("Datapoints", [])
            if dps:
                latest = sorted(dps, key=lambda d: d["Timestamp"])[-1]
                value = latest.get(stat) or latest.get("ExtendedStatistics", {}).get(stat)
                results[f"{metric_name}_{stat}"] = value
            else:
                results[f"{metric_name}_{stat}"] = None
        except Exception as e:
            results[f"{metric_name}_{stat}"] = f"ERROR: {e}"

    return results


def describe_endpoint_status(sm_client, endpoint_name: str) -> str:
    """Return the current endpoint status string."""
    try:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        return resp["EndpointStatus"]
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def microseconds_to_ms(us) -> str:
    if us is None:
        return "N/A"
    if isinstance(us, str):
        return us
    return f"{us / 1000:.2f} ms"


def _status_icon(status: str) -> str:
    return {"InService": "✅", "Updating": "🔄", "Failed": "❌"}.get(status, "⚠️")


def print_snapshot(label: str, status: str, metrics: dict):
    icon = _status_icon(status)
    print(f"\n  {icon}  {label}  [{status}]")
    print(f"     Invocations (sum)    : {metrics.get('Invocations_Sum', 'N/A')}")
    print(f"     Latency avg          : {microseconds_to_ms(metrics.get('ModelLatency_Average'))}")
    print(f"     Latency p90          : {microseconds_to_ms(metrics.get('ModelLatency_p90'))}")
    print(f"     Latency p99          : {microseconds_to_ms(metrics.get('ModelLatency_p99'))}")
    print(f"     4xx errors           : {metrics.get('Invocation4XXErrors_Sum', 'N/A')}")
    print(f"     5xx errors           : {metrics.get('Invocation5XXErrors_Sum', 'N/A')}")


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def poll_once(cfg: dict, region: str, log_file, dry_run: bool) -> dict:
    """Single poll iteration. Returns the snapshot dict."""
    ts = datetime.now(timezone.utc).isoformat()

    if dry_run:
        print(f"[DRY-RUN] Would query CloudWatch in region={region} at {ts}")
        return {"dry_run": True, "timestamp": ts}

    import boto3
    sm  = boto3.client("sagemaker",    region_name=region)
    cw  = boto3.client("cloudwatch",   region_name=region)

    lookback = cfg["monitor"]["cloudwatch_lookback_minutes"]
    snapshot = {"timestamp": ts, "endpoints": {}}

    for key in cfg["endpoints"].keys():
        ep_name  = cfg["endpoints"][key]["name"]
        variant  = cfg["endpoints"][key]["variant"]
        status   = describe_endpoint_status(sm, ep_name)
        metrics  = fetch_cloudwatch_metrics(cw, ep_name, variant, lookback)

        snapshot["endpoints"][key] = {
            "endpoint": ep_name,
            "status":   status,
            "metrics":  metrics,
        }
        print_snapshot(ep_name, status, metrics)

    # Threshold alerts (informational — alert_manager.py handles SNS)
    _check_thresholds(cfg, snapshot)

    # Log to JSONL
    if log_file:
        with open(log_file, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

    return snapshot


def _check_thresholds(cfg: dict, snapshot: dict):
    """Print warnings when metrics exceed configured thresholds."""
    t = cfg["thresholds"]
    lat_warn   = t["latency_ms"]["warning"]   * 1000  # convert ms → µs
    lat_crit   = t["latency_ms"]["critical"]  * 1000
    err_warn   = t["error_rate_pct"]["warning"]
    err_crit   = t["error_rate_pct"]["critical"]

    for key, ep_data in snapshot.get("endpoints", {}).items():
        m    = ep_data.get("metrics", {})
        name = ep_data["endpoint"]

        # Latency p99
        lat = m.get("ModelLatency_p99")
        if isinstance(lat, (int, float)):
            lat_ms = lat / 1000
            if lat_ms >= t["latency_ms"]["critical"]:
                print(f"  ⚠️  CRITICAL  [{name}] latency p99 = {lat_ms:.0f} ms (>{t['latency_ms']['critical']} ms)")
            elif lat_ms >= t["latency_ms"]["warning"]:
                print(f"  ⚠️  WARNING   [{name}] latency p99 = {lat_ms:.0f} ms (>{t['latency_ms']['warning']} ms)")

        # 5xx errors
        err5 = m.get("Invocation5XXErrors_Sum")
        inv  = m.get("Invocations_Sum") or 1
        if isinstance(err5, (int, float)) and err5 > 0:
            rate = (err5 / inv) * 100
            if rate >= err_crit:
                print(f"  ⚠️  CRITICAL  [{name}] 5xx error rate = {rate:.1f}% (>{err_crit}%)")
            elif rate >= err_warn:
                print(f"  ⚠️  WARNING   [{name}] 5xx error rate = {rate:.1f}% (>{err_warn}%)")

        # Endpoint not InService
        status = ep_data.get("status", "")
        if status not in ("InService", "Updating"):
            print(f"  🚨 ALERT     [{name}] endpoint status = {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS SageMaker endpoint monitor")
    parser.add_argument("--config",   default=str(Path(__file__).parent / "config.json"),
                        help="Path to config.json")
    parser.add_argument("--region",   default=None, help="AWS region (overrides config)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Poll interval in seconds (0 = once; default: from config)")
    parser.add_argument("--log-file", default=None,
                        help="JSONL file to append results (default: from config)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print actions without calling AWS")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    region = args.region or cfg["aws"]["region"]

    # Resolve log file
    if args.log_file:
        log_file = args.log_file
    else:
        log_dir  = Path(__file__).parent / cfg["monitor"]["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / cfg["monitor"]["log_filename"])

    interval = args.interval if args.interval is not None else cfg["monitor"]["poll_interval_seconds"]

    print("=" * 60)
    print("NIDS ENDPOINT MONITOR")
    print("=" * 60)
    print(f"Region   : {region}")
    print(f"Interval : {interval}s  (0 = run once)")
    print(f"Log file : {log_file}")
    if args.dry_run:
        print("Mode     : DRY-RUN")
    print("=" * 60)

    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Polling ...")
        poll_once(cfg, region, log_file, dry_run=args.dry_run)

        if interval == 0:
            break

        time.sleep(interval)


if __name__ == "__main__":
    main()
