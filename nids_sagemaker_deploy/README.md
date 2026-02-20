# SageMaker Model Deployment

This directory contains scripts for deploying the NIDS models to AWS SageMaker.

## Directory Structure

```
sagemaker_deploy/
├── README.md                      # This file
├── deploy_autoencoder.py          # Deploy autoencoder to SageMaker
├── deploy_xgboost.py              # Deploy XGBoost to SageMaker
├── inference_autoencoder.py       # Custom inference handler for autoencoder
├── inference_xgboost.py           # Custom inference handler for XGBoost
├── requirements_autoencoder.txt   # Dependencies for autoencoder endpoint
├── requirements_xgboost.txt       # Dependencies for XGBoost endpoint
└── config.json                    # SageMaker deployment configuration
```

## Prerequisites

1. **AWS Credentials**: Configure AWS CLI with appropriate credentials
   ```bash
   aws configure
   ```

2. **IAM Role**: Create SageMaker execution role with permissions:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`

3. **Python Dependencies**:
   ```bash
   pip install boto3 sagemaker torch xgboost
   ```

## Deployment Steps

### 1. Deploy Autoencoder

```bash
python deploy_autoencoder.py \
  --model-path ../nids_train/model/autoencoder.pt \
  --scaler-path ../nids_train/scaler_params.json \
  --endpoint-name nids-autoencoder \
  --instance-type ml.t2.medium
```

This will:
- Package the PyTorch model with custom inference script
- Upload to S3
- Create SageMaker model
- Deploy to endpoint

### 2. Deploy XGBoost

```bash
python deploy_xgboost.py \
  --model-path ../nids_train/model/xgb_classifier.onnx \
  --label-map-path ../ai_nids_rust/models/xgb_label_map.json \
  --endpoint-name nids-xgboost \
  --instance-type ml.t2.medium
```

### 3. Test Endpoints

```bash
# Test autoencoder
python test_endpoints.py --endpoint nids-autoencoder --test-type autoencoder

# Test XGBoost
python test_endpoints.py --endpoint nids-xgboost --test-type xgboost
```

## Endpoint Configuration

Edit `config.json` to customize:
- Instance types
- Endpoint names
- Auto-scaling policies
- Model data compression

## Monitoring

View endpoint metrics in AWS Console:
- CloudWatch Metrics: Invocations, latency, errors
- SageMaker Console: Endpoint status, instance health

## Cleanup

To delete endpoints and save costs:
```bash
python cleanup.py --endpoint-name nids-autoencoder
python cleanup.py --endpoint-name nids-xgboost
```

## Cost Estimation

- **ml.t2.medium**: ~$0.065/hour
- **ml.m5.large**: ~$0.115/hour
- **ml.c5.xlarge**: ~$0.204/hour

For demo purposes, use `ml.t2.medium` to minimize costs.
