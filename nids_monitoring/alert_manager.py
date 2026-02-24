"""
NIDS Alert Manager
===================
Checks endpoint metrics against configured thresholds and sends SNS
notifications when limits are breached. Uses a per-metric cooldown to
prevent alert storms.

Usage:
    python alert_manager.py [OPTIONS]

Options:
    --config PATH       Path to config.json (default: ./config.json)
    --region REGION     AWS region (overrides config)
    --test-alert        Publish a test SNS notification and exit
    --dry-run           Print alerts without calling SNS
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cooldown state (in-process, keyed by metric name)
# ---------------------------------------------------------------------------

_last_alert_times: dict = {}


def _is_in_cooldown(key: str, cooldown_secs: int) -> bool:
    last = _last_alert_times.get(key)
    if last is None:
        return False
    return (time.time() - last) < cooldown_secs


def _record_alert(key: str):
    _last_alert_times[key] = time.time()


# ---------------------------------------------------------------------------
# SNS publish
# ---------------------------------------------------------------------------

def publish_sns(topic_arn: str, subject: str, message: str, region: str, dry_run: bool):
    """Publish an alert message to an SNS topic."""
    if dry_run:
        print(f"[DRY-RUN] Would publish SNS alert:")
        print(f"  Topic  : {topic_arn}")
        print(f"  Subject: {subject}")
        print(f"  Body   : {message}")
        return

    try:
        import boto3
        sns = boto3.client("sns", region_name=region)
        sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)
        print(f"  📣 SNS alert sent: {subject}")
    except Exception as e:
        print(f"  ❌ SNS publish failed: {e}")


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------

def evaluate_snapshot(snapshot: dict, cfg: dict, region: str, dry_run: bool):
    """
    Check a monitor snapshot (from monitor_endpoints.py) against thresholds.
    Fires alerts for any breaches not in cooldown.
    """
    t          = cfg["thresholds"]
    cooldown   = cfg["alerts"]["cooldown_seconds"]
    topic_arn  = cfg["aws"].get("sns_topic_arn", "")
    enable_sns = cfg["alerts"]["enable_sns"] and bool(topic_arn)

    for key, ep_data in snapshot.get("endpoints", {}).items():
        name    = ep_data["endpoint"]
        status  = ep_data.get("status", "")
        metrics = ep_data.get("metrics", {})

        alerts = []

        # --- Endpoint status ---
        if status not in ("InService", "Updating"):
            alerts.append({
                "key":     f"{name}_status",
                "subject": f"[NIDS ALERT] Endpoint '{name}' is {status}",
                "body":    (
                    f"NIDS endpoint '{name}' has status '{status}'.\n"
                    f"Timestamp: {snapshot.get('timestamp')}\n"
                    f"Please check the AWS SageMaker console."
                ),
                "level": "CRITICAL",
            })

        # --- Latency p99 ---
        lat_us = metrics.get("ModelLatency_p99")
        if isinstance(lat_us, (int, float)):
            lat_ms = lat_us / 1000
            if lat_ms >= t["latency_ms"]["critical"]:
                alerts.append({
                    "key":     f"{name}_latency_critical",
                    "subject": f"[NIDS CRITICAL] High latency on '{name}': {lat_ms:.0f} ms",
                    "body":    (
                        f"Endpoint '{name}' p99 latency = {lat_ms:.0f} ms "
                        f"(threshold: {t['latency_ms']['critical']} ms).\n"
                        f"Timestamp: {snapshot.get('timestamp')}"
                    ),
                    "level": "CRITICAL",
                })
            elif lat_ms >= t["latency_ms"]["warning"]:
                alerts.append({
                    "key":     f"{name}_latency_warning",
                    "subject": f"[NIDS WARNING] Elevated latency on '{name}': {lat_ms:.0f} ms",
                    "body":    (
                        f"Endpoint '{name}' p99 latency = {lat_ms:.0f} ms "
                        f"(threshold: {t['latency_ms']['warning']} ms).\n"
                        f"Timestamp: {snapshot.get('timestamp')}"
                    ),
                    "level": "WARNING",
                })

        # --- 5xx error rate ---
        err5 = metrics.get("Invocation5XXErrors_Sum")
        inv  = metrics.get("Invocations_Sum") or 1
        if isinstance(err5, (int, float)) and err5 > 0:
            rate = (err5 / inv) * 100
            level_key = "critical" if rate >= t["error_rate_pct"]["critical"] else "warning"
            thresh     = t["error_rate_pct"][level_key]
            if rate >= thresh:
                alerts.append({
                    "key":     f"{name}_error_rate_{level_key}",
                    "subject": f"[NIDS {level_key.upper()}] 5xx errors on '{name}': {rate:.1f}%",
                    "body":    (
                        f"Endpoint '{name}' 5xx error rate = {rate:.1f}% "
                        f"(threshold: {thresh}%).\n"
                        f"Errors: {int(err5)}, Invocations: {int(inv)}\n"
                        f"Timestamp: {snapshot.get('timestamp')}"
                    ),
                    "level": level_key.upper(),
                })

        # --- Fire alerts ---
        for alert in alerts:
            akey = alert["key"]
            if _is_in_cooldown(akey, cooldown):
                print(f"  ⏸  [{alert['level']}] '{akey}' suppressed (cooldown {cooldown}s)")
                continue

            _record_alert(akey)
            print(f"  🚨 [{alert['level']}] {alert['subject']}")

            if enable_sns:
                publish_sns(topic_arn, alert["subject"], alert["body"], region, dry_run)
            elif dry_run:
                print(f"     [DRY-RUN] SNS disabled or no topic ARN configured.")


# ---------------------------------------------------------------------------
# Standalone run: reads latest JSONL log entry and evaluates
# ---------------------------------------------------------------------------

def evaluate_latest_log(log_path: str, cfg: dict, region: str, dry_run: bool):
    """Read last line of a monitor JSONL log and evaluate alerts."""
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {log_path}")
        return

    last_line = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        print("Log file is empty.")
        return

    snapshot = json.loads(last_line)
    ts = snapshot.get("timestamp", "unknown")
    print(f"Evaluating snapshot from: {ts}")
    evaluate_snapshot(snapshot, cfg, region, dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS alert manager")
    parser.add_argument("--config",      default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--region",      default=None)
    parser.add_argument("--test-alert",  action="store_true",
                        help="Publish a test SNS alert and exit")
    parser.add_argument("--log-file",    default=None,
                        help="Evaluate latest entry in a monitor JSONL log")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    region = args.region or cfg["aws"]["region"]

    print("=" * 60)
    print("NIDS ALERT MANAGER")
    print("=" * 60)

    if args.test_alert:
        topic_arn = cfg["aws"].get("sns_topic_arn", "")
        if not topic_arn and not args.dry_run:
            print("No SNS topic ARN configured in config.json → aws.sns_topic_arn")
            return
        publish_sns(
            topic_arn=topic_arn,
            subject="[NIDS TEST] Alert manager test notification",
            message=(
                "This is a test alert from the NIDS monitoring system.\n"
                f"Timestamp : {datetime.now(timezone.utc).isoformat()}\n"
                f"Region    : {region}"
            ),
            region=region,
            dry_run=args.dry_run,
        )
        return

    log_file = args.log_file
    if not log_file:
        log_dir  = Path(__file__).parent / cfg["monitor"]["log_dir"]
        log_file = str(log_dir / cfg["monitor"]["log_filename"])

    evaluate_latest_log(log_file, cfg, region, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
