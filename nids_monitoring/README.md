# NIDS Monitoring

Monitoring scripts for the two deployed SageMaker endpoints:
- **`nids-autoencoder`** — anomaly detection (MSE reconstruction error)
- **`nids-xgboost`** — attack classification

## Directory Structure

```
nids_monitoring/
├── config.json              # All thresholds, endpoint names, AWS settings
├── monitor_endpoints.py     # CloudWatch metric poller
├── alert_manager.py         # Threshold alerts + SNS notifications
├── drift_detector.py        # MSE statistical drift detection
├── dashboard.py             # Live terminal dashboard
├── requirements.txt
└── README.md
```

## Prerequisites

1. **AWS credentials** configured (`aws configure` or env vars)
2. **IAM permissions** — the identity running these scripts needs:
   - `cloudwatch:GetMetricStatistics`
   - `sagemaker:DescribeEndpoint`, `sagemaker:ListEndpoints`
   - `sns:Publish` *(only if SNS alerting is enabled)*

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.json` to customise:

| Field | Description |
|---|---|
| `endpoints.*.name` | SageMaker endpoint names (defaults match deployment) |
| `aws.region` | AWS region |
| `aws.sns_topic_arn` | SNS topic for alerts — leave blank to disable |
| `thresholds.mse.*` | p95/p99/p99.9 MSE cut-offs (from training eval) |
| `thresholds.latency_ms.*` | Warning/critical latency limits |
| `thresholds.error_rate_pct.*` | 5xx error rate limits |
| `monitor.poll_interval_seconds` | How often to poll CloudWatch |
| `alerts.cooldown_seconds` | Minimum gap between repeat alerts |

---

## Usage

### 1. Endpoint Monitor (CloudWatch poller)

Polls both endpoints on a configurable interval and logs to JSONL:

```bash
# Run once (interval = 0)
python monitor_endpoints.py --interval 0 --region us-east-1

# Poll every 60 s (default) and write to logs/monitor.jsonl
python monitor_endpoints.py --region us-east-1

# Dry-run (no AWS calls)
python monitor_endpoints.py --dry-run
```

The log file (`logs/monitor.jsonl`) is append-only — one JSON object per poll.

---

### 2. Alert Manager

Reads the latest monitor log entry and fires alerts on threshold breaches:

```bash
# Evaluate latest log snapshot and print alerts
python alert_manager.py

# Send a test SNS notification (requires sns_topic_arn in config.json)
python alert_manager.py --test-alert --region us-east-1

# Dry-run (compute + print, no SNS call)
python alert_manager.py --dry-run
```

**To enable SNS alerts**, set `aws.sns_topic_arn` in `config.json` and set `alerts.enable_sns` to `true`.

---

### 3. Drift Detector

Detect statistical drift in autoencoder MSE reconstruction error:

```bash
# From a CSV file of MSE values (e.g. from Rust client output)
python drift_detector.py --input-csv path/to/mse_values.csv --mse-column mse_error

# From the monitor log (uses latency as MSE proxy)
python drift_detector.py --from-logs logs/monitor.jsonl

# Dry-run with synthetic data
python drift_detector.py --dry-run

# Emit result as a custom CloudWatch metric (namespace: NIDS/DriftDetection)
python drift_detector.py --input-csv mse_values.csv --emit-cloudwatch
```

**CSV format** — any CSV with a numeric column works:
```csv
timestamp,mse_error
2025-01-01T00:00:00,0.0182
2025-01-01T00:01:00,0.0211
...
```

A `drift_report.json` is written on every run. The script exits with code `1`
when severity is HIGH or CRITICAL (useful for CI/cron-based monitoring).

---

### 4. Live Dashboard

Terminal dashboard refreshing every N seconds:

```bash
# Real data
python dashboard.py --region us-east-1 --refresh 30

# Dry-run with mock data (no AWS calls needed)
python dashboard.py --dry-run
```

Example output:
```
╔══════════════════════════════════════════════════════════════╗
║              NIDS MONITORING DASHBOARD                       ║
║  2025-01-15 09:42:01  region: us-east-1                      ║
╠══════════════════════════════════════════════════════════════╣
║  🟢  nids-autoencoder                                        ║
║      Status     : InService                                  ║
║      Invocations: 142                                        ║
║      Latency p99: 87.2 ms  🟢  [██░░░░░░░░░░░░░░░░░░░░]    ║
╠──────────────────────────────────────────────────────────────╣
║  🟢  nids-xgboost                                            ║
║      ...                                                     ║
╠══════════════════════════════════════════════════════════════╣
║  📊 Drift Detection (autoencoder MSE)                        ║
║      Severity  : 🟢  OK                                      ║
║      Mean MSE  : 0.018400  (p95 threshold: 0.0234)           ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Typical Workflow

```bash
# Terminal 1: continuous monitor (writes logs)
python monitor_endpoints.py --region us-east-1

# Terminal 2: live dashboard
python dashboard.py --region us-east-1

# On demand: check for drift after a test run
python drift_detector.py --input-csv mse_results.csv

# On demand: check if any alerts need to fire
python alert_manager.py
```

---

## Connecting to the Deployment

These scripts use the same endpoint names as `nids_sagemaker_deploy/config.json`.
After running `deploy_xgboost.py` and `deploy_autoencoder.py`, the endpoints
will be live and these scripts will start collecting metrics immediately.
