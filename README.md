# AAI-540-Group-11: AI-Powered Network Intrusion Detection (NIDS)

This repository contains a full-stack AI-powered Network Intrusion Detection System (NIDS) and User and Entity Behavior Analytics (UEBA) platform. This project integrates machine learning models into a real-time monitoring and defensive pipeline deployed on AWS.

## 🔐 NIDS Stack Overview
The NIDS component uses a hybrid machine learning approach:
- **Autoencoder (Anomaly Detection)**: Benchmarks "normal" network traffic and flags deviations.
- **XGBoost (Attack Classification)**: Classifies flagged traffic into specific attack types (DoS, PortScan, etc.).
- **Rust Engine**: A high-performance inference engine that connects the models to live traffic.

### 📖 Step-by-Step Instructions
For a detailed guide on training, deploying, and testing the NIDS stack, please see:
👉 **[NIDS Step-by-Step Instructions](nids_instructions.md)**

---

## 📂 Repository Structure
- `nids_train/`: Model training scripts and preprocessing.
- `ai_nids_rust/`: High-performance Rust inference engine.
- `nids_sagemaker_deploy/`: AWS SageMaker deployment scripts for XGBoost.
- `nids_lambda_actions/`: Automated defensive actions (Lambda).
- `nids_monitoring/`: Dashboards and performance metrics.
- `ueba_*/`: User and Entity Behavior Analytics components (WIP).

## 🚀 Quick Start (Deployment)
To deploy the entire NIDS stack to AWS:
```bash
./deploy_all.sh
```
*Requires AWS credentials and Zig established locally.*

---

## 👥 Contributors
Developed by Group 11 for AAI-540.
