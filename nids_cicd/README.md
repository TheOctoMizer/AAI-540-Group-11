# NIDS CI/CD

Automates the **retrain → evaluate → gate → redeploy** cycle for both NIDS models.

```
nids_cicd/
├── config.json                         # Gates, endpoint names, AWS settings
├── pipeline.py                         # Main pipeline orchestrator (8 steps)
├── evaluate.py                         # Standalone quality-gate checker
├── requirements.txt
├── .github/workflows/nids_cicd.yml     # GitHub Actions workflow
└── README.md
```

## Pipeline Steps

```
1. train_autoencoder   → calls nids_train/train_ae.py
2. eval_autoencoder    → MSE mean / p95 / p99 quality gates
3. train_xgboost       → calls nids_train/train_xgb.py (uses AE encoding)
4. eval_xgboost        → accuracy gate + ONNX consistency check
5. upload_artifacts    → pushes .pt and .onnx files to S3
6. deploy_autoencoder  → calls nids_sagemaker_deploy/deploy_autoencoder.py
7. deploy_xgboost      → calls nids_sagemaker_deploy/deploy_xgboost.py
8. smoke_test          → invokes both live endpoints with test payloads
```

**If any gate fails, the pipeline halts immediately and does not deploy.**

## Prerequisites

1. **Python 3.10+** with the project virtualenv:
   ```bash
   pip install -r nids_cicd/requirements.txt
   ```

2. **AWS credentials** configured (`aws configure`, or env vars)

3. **IAM permissions** needed:
   - `sagemaker:CreateModel`, `CreateEndpoint*`, `UpdateEndpoint`, `DescribeEndpoint`
   - `s3:PutObject`
   - `iam:PassRole` (to pass the SageMaker execution role)

## Running Locally

```bash
cd nids_cicd

# Full pipeline (retrain + gate-check + deploy)
python3 pipeline.py \
  --role   arn:aws:iam::ACCOUNT:role/LabRole \
  --bucket nids-mlops-models \
  --region us-east-1

# Gate-check only (skip retrain + deploy) — fast
python3 pipeline.py --skip-train --skip-deploy

# Dry-run — no AWS calls, uses synthetic data
python3 pipeline.py --dry-run

# Standalone evaluation (no retraining)
python3 evaluate.py --dry-run
python3 evaluate.py --model-dir ../nids_train/model
```

## GitHub Actions

The workflow triggers automatically on pushes to `main` that modify training or
deployment code, or manually via **Actions → Run workflow**.

### Required Repository Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_SESSION_TOKEN` | Session token (for lab/temporary credentials) |
| `AWS_REGION` | AWS region (default: `us-east-1`) |
| `SAGEMAKER_ROLE_ARN` | Full ARN of the SageMaker execution role |
| `S3_BUCKET` | S3 bucket for model artifacts |

Add these at **Settings → Secrets and variables → Actions**.

### Manual Trigger Options

| Input | Default | Description |
|---|---|---|
| `skip_train` | `false` | Skip retraining, use existing artifacts |
| `skip_deploy` | `false` | Evaluate only, do not redeploy |
| `dry_run` | `false` | No AWS calls — useful for testing the workflow |

## Quality Gates

Configured in `config.json → quality_gates`:

| Model | Gate | Default Threshold |
|---|---|---|
| Autoencoder | MSE mean | ≤ 0.05 |
| Autoencoder | MSE p95 | ≤ 0.0234 |
| Autoencoder | MSE p99 | ≤ 0.0567 |
| XGBoost | Accuracy | ≥ 0.92 |
| XGBoost | ONNX max diff | ≤ 1e-5 |
| XGBoost | ONNX mean diff | ≤ 1e-6 |

> The MSE thresholds are shared with `nids_monitoring/config.json` — keep them in sync.

## cicd_report.json

Every pipeline run writes `cicd_report.json`:

```json
{
  "timestamp": "2025-01-15T09:42:00Z",
  "overall_passed": true,
  "dry_run": false,
  "steps": [
    {"name": "train_autoencoder", "passed": true, "elapsed": 312.4, ...},
    {"name": "eval_autoencoder",  "passed": true, "elapsed": 1.2,
     "gates": {"mse_p95": {"value": 0.0213, "threshold": 0.0234, "passed": true}}},
    ...
  ]
}
```

In GitHub Actions this is uploaded as a workflow artifact for every run
(retained 30 days), even if the pipeline fails.
