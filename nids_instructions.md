# NIDS: Step-by-Step Instructions

This guide provides a comprehensive walkthrough for the Network Intrusion Detection System (NIDS) lifecycle, from data training to cloud deployment.

## 🏗️ Architecture Overview
The NIDS stack consists of:
1.  **Training Pipeline**: Autoencoder (anomaly detection) + XGBoost (attack classification).
2.  **Rust Engine**: Real-time inference engine deployed on EC2 (ARM64).
3.  **SageMaker Endpoint**: AWS-hosted XGBoost model for secondary classification.
4.  **Go Target Server**: A simulated target server for monitoring traffic.
5.  **Lambda Shutdown**: Automatic defensive action to stop compromised instances.

---

## 🚀 Phase 1: Model Training & Export
All training happens in the `nids_train/` directory.

### 1. Setup Environment
```bash
cd nids_train
pip install -r requirements.txt # or install torch, pandas, onnxruntime, scikit-learn
```

### 2. Train the Autoencoder
Train on benign traffic (`Monday-WorkingHours.pcap_ISCX.csv`) to establish a "normal" baseline.
```bash
python train.py --dataset ./nids_dataset/Monday-WorkingHours.pcap_ISCX.csv
```
**Assets Generated:**
- `nids_autoencoder.onnx`: Exported for the Rust Engine.
- `scaler_params.json`: Feature scaling parameters.
- `detection_threshold.json`: Statistical thresholds for anomaly detection.

### 3. Train the XGBoost Classifier
Train on attack traffic (`Wednesday-WorkingHours.pcap_ISCX.csv`) to classify detection types.
```bash
python train_xgb.py
```
**Assets Generated:**
- `xgb_classifier.onnx`: For SageMaker deployment.
- `xgb_label_map.json`: Maps numeric outputs to attack names (e.g., DoS, PortScan).

---

## ☁️ Phase 2: AWS Infrastructure Deployment
Use the consolidated script `deploy_all.sh` to launch the entire stack.

### 1. Prerequisites
- AWS CLI configured with valid credentials.
- `zig` installed (for local cross-compilation of the Rust binary).
- S3 bucket named `nids-mlops-models` (or update the script).

### 2. Run Deployment
```bash
./deploy_all.sh
```
**What this script does:**
1.  **Lambda**: Deploys `nids-vm-shutdown` for automated defense.
2.  **Go Server**: Launches a `t4g.micro` EC2 in Subnet A.
3.  **SageMaker**: Packages XGBoost ONNX and deploys an endpoint (`ml.m5.large`).
4.  **Rust NIDS**: 
    - Cross-compiles the Rust engine locally for Linux ARM64.
    - Launches a `t4g.small` EC2 in Subnet B.
    - Copies binary/models and starts the `nids.service`.

---

## 🛠️ Phase 3: Running & Testing the NIDS

### 1. Monitor Logs
SSH into the Rust NIDS instance (IP found in `deployment_info.json`) and check the service:
```bash
sudo journalctl -u nids -f
```

### 2. Trigger an Inference Test
Manually trigger the NIDS engine to process a batch of data:
```bash
curl -X POST http://<NIDS_PUBLIC_IP>:3000/trigger
```

### 3. Verify Automatic Shutdown
If a high-confidence attack (e.g., Heartbleed) is detected:
1.  The Rust engine invokes the **Lambda** function.
2.  The Lambda function stops the **Go Server** instance.
3.  Verify the instance state in the AWS Console.

---

## 📊 Phase 4: Monitoring & Dashboards
Use the `nids_monitoring/` tools to visualize performance.
```bash
cd nids_monitoring
python dashboard.py
```
*Note: Ensure you have access to the monitoring logs or CloudWatch metrics as configured.*
