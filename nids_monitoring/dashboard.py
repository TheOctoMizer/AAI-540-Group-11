"""
NIDS Live Terminal Dashboard
=============================
Refreshes the terminal every N seconds with endpoint health, recent CloudWatch
metrics, and the latest drift report.

Uses a plain polling loop (no ncurses dependency) — clears the screen on each
refresh so it reads like a live dashboard.

Usage:
    python dashboard.py [OPTIONS]

Options:
    --config PATH       Path to config.json (default: ./config.json)
    --region REGION     AWS region (overrides config)
    --refresh SECS      Refresh interval in seconds (overrides config)
    --dry-run           Display mock data without calling AWS
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def bar(value, max_value, width=20, fill="█", empty="░") -> str:
    """Simple ASCII progress bar."""
    if max_value <= 0 or value is None:
        return empty * width
    ratio = min(value / max_value, 1.0)
    filled = int(ratio * width)
    return fill * filled + empty * (width - filled)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_cw_stat(cw, endpoint_name: str, variant: str, metric: str, stat: str, lookback_min: int):
    """Fetch latest CloudWatch data point; return None on error."""
    now   = datetime.now(timezone.utc)
    start = now - timedelta(minutes=lookback_min)
    try:
        is_percentile = stat.startswith("p")
        kwargs = {
            "Namespace": "AWS/SageMaker",
            "MetricName": metric,
            "Dimensions": [
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "VariantName",  "Value": variant},
            ],
            "StartTime": start,
            "EndTime": now,
            "Period": lookback_min * 60,
        }
        if is_percentile:
            kwargs["ExtendedStatistics"] = [stat]
        else:
            kwargs["Statistics"] = [stat]

        resp = cw.get_metric_statistics(**kwargs)
        dps = resp.get("Datapoints", [])
        if dps:
            latest = sorted(dps, key=lambda d: d["Timestamp"])[-1]
            return latest.get(stat) or latest.get("ExtendedStatistics", {}).get(stat)
    except Exception:
        pass
    return None


def fetch_endpoint_data(cfg: dict, region: str) -> dict:
    """Query SageMaker + CloudWatch for live data. Returns structured dict."""
    import boto3
    sm    = boto3.client("sagemaker",  region_name=region)
    cw    = boto3.client("cloudwatch", region_name=region)

    lookback = cfg["monitor"]["cloudwatch_lookback_minutes"]
    data     = {}

    for key in cfg["endpoints"].keys():
        ep_name = cfg["endpoints"][key]["name"]
        variant = cfg["endpoints"][key]["variant"]

        try:
            status = sm.describe_endpoint(EndpointName=ep_name)["EndpointStatus"]
        except Exception as e:
            status = f"ERROR ({e})"

        invocations = _fetch_cw_stat(cw, ep_name, variant, "Invocations",         "Sum",  lookback)
        lat_avg     = _fetch_cw_stat(cw, ep_name, variant, "ModelLatency",        "Average", lookback)
        lat_p99     = _fetch_cw_stat(cw, ep_name, variant, "ModelLatency",        "p99",  lookback)
        err4xx      = _fetch_cw_stat(cw, ep_name, variant, "Invocation4XXErrors", "Sum",  lookback)
        err5xx      = _fetch_cw_stat(cw, ep_name, variant, "Invocation5XXErrors", "Sum",  lookback)

        data[key] = {
            "endpoint":    ep_name,
            "status":      status,
            "invocations": invocations,
            "lat_avg_ms":  (lat_avg / 1000) if isinstance(lat_avg, (int, float)) else None,
            "lat_p99_ms":  (lat_p99 / 1000) if isinstance(lat_p99, (int, float)) else None,
            "err4xx":      err4xx,
            "err5xx":      err5xx,
        }

    return data


def mock_endpoint_data(cfg: dict) -> dict:
    """Return plausible mock data for dry-run mode."""
    data = {}
    if "autoencoder" in cfg["endpoints"]:
        data["autoencoder"] = {
            "endpoint":    cfg["endpoints"]["autoencoder"]["name"],
            "status":      "InService",
            "invocations": 142,
            "lat_avg_ms":  18.4,
            "lat_p99_ms":  87.2,
            "err4xx":      0,
            "err5xx":      0,
        }
    if "xgboost" in cfg["endpoints"]:
        data["xgboost"] = {
            "endpoint":    cfg["endpoints"]["xgboost"]["name"],
            "status":      "InService",
            "invocations": 142,
            "lat_avg_ms":  5.1,
            "lat_p99_ms":  22.9,
            "err4xx":      0,
            "err5xx":      0,
        }
    return data


def load_drift_report(cfg: dict) -> dict | None:
    """Load the latest drift report if it exists."""
    report_path = Path(__file__).parent / cfg["drift"]["output_report"]
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

STATUS_ICON = {
    "InService": "🟢",
    "Updating":  "🔵",
    "Creating":  "🔵",
    "Failed":    "🔴",
}


def ms_str(v) -> str:
    return f"{v:.1f} ms" if isinstance(v, (int, float)) else "—"


def render_dashboard(cfg: dict, ep_data: dict, drift: dict | None, region: str, dry_run: bool):
    W = 62
    t = cfg["thresholds"]

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("╔" + "═" * W + "╗")
    title = " NIDS MONITORING DASHBOARD "
    pad   = (W - len(title)) // 2
    print("║" + " " * pad + title + " " * (W - pad - len(title)) + "║")
    ts_line = f" {now_str}  region: {region}" + ("  [DRY-RUN]" if dry_run else "")
    print("║" + ts_line.ljust(W) + "║")
    print("╠" + "═" * W + "╣")

    ep_keys = list(ep_data.keys())
    for i, key in enumerate(ep_keys):
        d    = ep_data[key]
        icon = STATUS_ICON.get(d["status"], "⚪")
        name = d["endpoint"]

        print("║" + f"  {icon}  {name}".ljust(W) + "║")
        print("║" + f"      Status     : {d['status']}".ljust(W) + "║")

        # Invocations
        inv = d["invocations"]
        print("║" + f"      Invocations: {inv if inv is not None else '—'}".ljust(W) + "║")

        # Latency
        lat_p99  = d["lat_p99_ms"]
        lat_crit = t["latency_ms"]["critical"]
        lat_warn = t["latency_ms"]["warning"]
        lat_icon = ("🔴" if isinstance(lat_p99, float) and lat_p99 >= lat_crit
                    else "🟡" if isinstance(lat_p99, float) and lat_p99 >= lat_warn
                    else "🟢")
        lat_bar  = bar(lat_p99 or 0, lat_crit)
        print("║" + f"      Latency avg : {ms_str(d['lat_avg_ms'])}".ljust(W) + "║")
        print("║" + f"      Latency p99 : {ms_str(lat_p99)}  {lat_icon}  [{lat_bar}]".ljust(W) + "║")

        # Errors
        e4 = d["err4xx"] or 0
        e5 = d["err5xx"] or 0
        err_icon = "🔴" if e5 > 0 else "🟢"
        print("║" + f"      Errors 4xx/5xx: {int(e4)}/{int(e5)}  {err_icon}".ljust(W) + "║")

        if i < len(ep_keys) - 1:
            print("╠" + "─" * W + "╣")

    print("╠" + "═" * W + "╣")

    # Drift section
    if drift:
        sev      = drift.get("severity", "OK")
        sev_icon = {"OK": "🟢", "ELEVATED": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(sev, "⚪")
        ts_drift = drift.get("timestamp", "")[:19].replace("T", " ")
        mean_mse = drift.get("overall_mean", 0)
        drift_n  = drift.get("drift_event_count", 0)
        samples  = drift.get("samples", 0)

        print("║" + "  📊 Drift Detection (autoencoder MSE)".ljust(W) + "║")
        print("║" + f"      Severity  : {sev_icon}  {sev}".ljust(W) + "║")
        print("║" + f"      Mean MSE  : {mean_mse:.6f}  (p95 threshold: {t['mse']['p95']})".ljust(W) + "║")
        print("║" + f"      Drift events: {drift_n}/{samples}  ({drift.get('drift_rate_pct', 0):.1f}%)".ljust(W) + "║")
        print("║" + f"      Last run  : {ts_drift}".ljust(W) + "║")
    else:
        print("║" + "  📊 Drift Detection: no report found".ljust(W) + "║")
        print("║" + "      Run: python drift_detector.py --dry-run".ljust(W) + "║")

    print("╚" + "═" * W + "╝")
    print(f"\n  Press Ctrl+C to exit.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS live terminal dashboard")
    parser.add_argument("--config",  default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--region",  default=None)
    parser.add_argument("--refresh", type=int, default=None,
                        help="Refresh interval in seconds (overrides config)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    region  = args.region or cfg["aws"]["region"]
    refresh = args.refresh if args.refresh is not None else cfg["dashboard"]["refresh_seconds"]

    while True:
        try:
            if args.dry_run:
                ep_data = mock_endpoint_data(cfg)
            else:
                ep_data = fetch_endpoint_data(cfg, region)

            drift = load_drift_report(cfg)

            clear_screen()
            print()
            render_dashboard(cfg, ep_data, drift, region, dry_run=args.dry_run)
            print(f"\n  Next refresh in {refresh}s ...")

        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

        time.sleep(refresh)


if __name__ == "__main__":
    main()
