# SageMaker CloudWatch Alarms for UEBA Model Monitoring

## Overview

This notebook sets up **automated CloudWatch Alarms** to monitor the health, performance, fairness, and explainability of the live UEBA (User and Entity Behavior Analytics) model deployed on SageMaker.

It creates proactive alerts for:
- Model performance degradation (F2 Score)
- Data quality issues (Feature Drift)
- Bias / Fairness violations (Disparate Impact)
- Explainability drift (SHAP-based logic changes)

All alarms are connected to **Amazon SNS** for instant notifications to the team.

**Notebook Title:**  
**SageMaker CloudWatch Alarms for UEBA Model Monitoring**

---

## Key Features

- High-priority F2 Score monitoring (weighted toward Recall)
- Feature Drift detection (Data Quality)
- Disparate Impact (DI) monitoring with dual thresholds (0.8 and 1.25)
- Explainability Drift monitoring using SHAP values
- Hourly evaluation periods with SNS notifications for both Alarm and OK states
- Clean, reusable alarm creation code for production use

---

## Notebook Structure

### I. High-Priority Model Performance Monitoring
- Creates alarm for F2 Score dropping below 0.8
- Focuses on Recall-heavy metric (critical for security/insider threat detection)

### II. Automated Data Quality and Feature Drift Monitoring
- Monitors `feature_baseline_drift` metric
- Uses Maximum statistic to catch any single feature drifting

### III. Automated Fairness and Bias Monitoring
- Dual alarms for Disparate Impact (DI):
  - Lower bound: DI < 0.8
  - Upper bound: DI > 1.25
- Ensures model does not unfairly favor or penalize any group

### IV. Automated Monitoring for Explainability Drift
- Monitors SHAP-based explainability drift
- Ensures the model’s decision logic remains stable over time

---

## Prerequisites

- SageMaker endpoint `ueba-endpoint2026216-v2` must be active and sending data
- Amazon SNS topic already created and subscribed (email/Slack/pager)
- IAM role with permissions to:
  - `cloudwatch:PutMetricAlarm`
  - `sns:Publish`
- SageMaker Model Monitor / Clarify jobs must be running to generate the required metrics

---

## How to Run

1. Open `sagemaker_alarms.ipynb`
2. Update the following variables:
   - `endpoint_name` (if different)
   - `sns_topic_arn` → Replace with your actual SNS topic ARN
3. Run all cells sequentially
4. Verify alarms appear in the CloudWatch console under **Alarms**

---

## Created Alarms

| Alarm Name                                | Metric                        | Threshold          | Purpose                              |
|-------------------------------------------|-------------------------------|--------------------|--------------------------------------|
| `ueba-endpoint2026216-v2-F2-Low`          | `f2`                          | < 0.8              | Model performance (Recall-focused)   |
| `ueba-endpoint2026216-v2-FeatureDrift`    | `feature_baseline_drift`      | ≥ 1.0              | Data Quality / Feature Drift         |
| `ueba-endpoint2026216-v2-DI-Low`          | `DI` (Disparate Impact)       | < 0.8              | Bias – Under-representation          |
| `ueba-endpoint2026216-v2-DI-High`         | `DI` (Disparate Impact)       | > 1.25             | Bias – Over-representation           |
| `ueba-endpoint2026216-v2-ExplainabilityDrift` | `shap_drift`               | ≥ 1.0              | Explainability / Logic Drift         |

---

## Significance for Project Governance

- Ensures **continuous reliability** of the UEBA anomaly detection system
- Provides early warning for model degradation, data drift, and bias
- Supports compliance, auditability, and responsible AI practices
- Enables proactive maintenance instead of reactive firefighting

---

## License

Part of AAI-540 Group 11 UEBA Project. All rights reserved.

---

**Last Updated:** February 2026