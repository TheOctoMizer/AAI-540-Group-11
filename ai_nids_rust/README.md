# NIDS Rust Server

AI-based Network Intrusion Detection System using SageMaker for inference.

## Architecture

```
Production CSV → /trigger → SageMaker Autoencoder → Threshold Check
                                ↓
                         If anomalous → SageMaker XGBoost
                                ↓
                         If malicious → Lambda → VM Shutdown
```

## Prerequisites

1. **AWS Credentials**: Configure AWS CLI
   ```bash
   aws configure
   ```

2. **SageMaker Endpoints**: Deploy models first
   ```bash
   cd ../sagemaker_deploy
   python deploy_autoencoder.py --model-path ../nids_train/model/autoencoder.pt --scaler-path ../nids_train/scaler_params.json
   python deploy_xgboost.py --model-path ../nids_train/model/xgb_classifier.onnx --label-map-path ./models/xgb_label_map.json
   ```

3. **Lambda Function**: Create VM shutdown function (Phase 5)

4. **Production Dataset**: Ensure dataset exists
   ```bash
   ls ../nids_train/output/features/production/
   # Should see: features.parquet, labels.csv, metadata.json
   ```

## Building

```bash
cargo build --release
```

## Running

```bash
# With default settings
cargo run --release

# With custom endpoints
cargo run --release -- \
  --autoencoder-endpoint nids-autoencoder \
  --xgboost-endpoint nids-xgboost \
  --lambda-function nids-vm-shutdown \
  --threshold 0.0567 \
  --log-level info
```

## Endpoints

### GET /health
Health check endpoint.

```bash
curl http://localhost:3000/health
```

### GET /ping
Middleware endpoint that forwards to Go server.

```bash
curl http://localhost:3000/ping
```

### POST /trigger
Trigger attack simulation from production dataset.

```bash
# Simulate 100 random samples
curl -X POST http://localhost:3000/trigger \
  -H "Content-Type: application/json" \
  -d '{}'

# Simulate specific attack type
curl -X POST http://localhost:3000/trigger \
  -H "Content-Type: application/json" \
  -d '{"attack_type": "DDoS", "count": 50}'

# Available attack types:
# - BENIGN
# - DDoS
# - PortScan
# - Bot
# - Infiltration
# - Web Attack – Brute Force
# - Web Attack – XSS
# - Web Attack – Sql Injection
```

**Response:**
```json
{
  "total_samples": 100,
  "benign_count": 45,
  "anomalous_count": 55,
  "malicious_count": 30,
  "lambda_triggered": true,
  "detections": [
    {
      "sample_index": 0,
      "true_label": "DDoS",
      "mse_error": 0.0892,
      "is_anomalous": true,
      "predicted_label": "DDoS",
      "confidence": 0.9876,
      "action": "MALICIOUS - DDoS"
    }
  ]
}
```

## Configuration

### Command-line Arguments

- `--autoencoder-endpoint`: SageMaker autoencoder endpoint name (default: `nids-autoencoder`)
- `--xgboost-endpoint`: SageMaker XGBoost endpoint name (default: `nids-xgboost`)
- `--lambda-function`: Lambda function name (default: `nids-vm-shutdown`)
- `--production-data-dir`: Path to production dataset (default: `../nids_train/output/features/production`)
- `--go-server-url`: Go server URL for ping middleware (default: `http://localhost:8080`)
- `--threshold`: MSE error threshold for anomaly detection (default: `0.0567` = p99)
- `--log-level`: Log level (default: `info`)

### Environment Variables

AWS credentials are loaded from:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

Or use AWS CLI configuration:
```bash
aws configure
```

## Detection Logic

1. **Load sample** from production dataset
2. **Call autoencoder** via SageMaker
   - Get reconstruction + MSE error
3. **Check threshold**
   - If MSE < threshold → BENIGN (skip XGBoost)
   - If MSE >= threshold → ANOMALOUS (proceed to step 4)
4. **Call XGBoost** via SageMaker
   - Get attack classification
5. **Check if malicious**
   - If label == "BENIGN" → ANOMALOUS but safe
   - If label != "BENIGN" → MALICIOUS (trigger Lambda)
6. **Trigger Lambda** (once per simulation)
   - Send attack info
   - Lambda shuts down target VM

## Testing

1. **Start Go server** (in another terminal):
   ```bash
   cd ../go_server
   go run main.go
   ```

2. **Start NIDS server**:
   ```bash
   cargo run --release
   ```

3. **Test health check**:
   ```bash
   curl http://localhost:3000/health
   ```

4. **Test ping middleware**:
   ```bash
   curl http://localhost:3000/ping
   # Should forward to Go server and return "pong"
   ```

5. **Trigger simulation**:
   ```bash
   curl -X POST http://localhost:3000/trigger \
     -H "Content-Type: application/json" \
     -d '{"attack_type": "DDoS", "count": 10}'
   ```

## Troubleshooting

### "Failed to invoke autoencoder endpoint"
- Ensure SageMaker endpoint is deployed and running
- Check AWS credentials
- Verify endpoint name matches

### "Failed to load production dataset"
- Run data pipeline first: `cd ../nids_train && python data_pipeline.py`
- Check path: `../nids_train/output/features/production/`

### "Failed to invoke Lambda function"
- Ensure Lambda function is created (Phase 5)
- Check IAM permissions
- Verify function name

## Cost Monitoring

- **SageMaker**: ~$0.13/hour (2 endpoints × ml.t2.medium)
- **Lambda**: ~$0.0000002 per invocation
- **Data Transfer**: Minimal for demo

**Tip**: Delete SageMaker endpoints after demo to save costs:
```bash
cd ../sagemaker_deploy
python cleanup.py --delete-all --yes
```
