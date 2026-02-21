"""
Deploy XGBoost classifier to AWS SageMaker.

This script:
1. Packages the ONNX model with custom inference code
2. Uploads to S3
3. Creates SageMaker model
4. Creates endpoint config and endpoint via boto3

No dependency on the sagemaker Python SDK — uses boto3 directly.
"""

import argparse
import json
import os
import tarfile
import time
import boto3
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Container image URIs for PyTorch inference (us-east-1).
# See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# ---------------------------------------------------------------------------
PYTORCH_INFERENCE_IMAGES = {
    "us-east-1":      "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",
    "us-west-2":      "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",
    "eu-west-1":      "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",
    "ap-southeast-1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",
    "ap-south-1":     "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",
}


def get_image_uri(region: str) -> str:
    uri = PYTORCH_INFERENCE_IMAGES.get(region)
    if not uri:
        # Fall back: substitute the region into the us-east-1 pattern
        uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.0-cpu-py310"
        print(f"  [WARN] Region '{region}' not in known list — using inferred URI: {uri}")
    return uri


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def package_model(model_path: str, label_map_path: str, inference_script: str, output_dir: str) -> str:
    """
    Package model artifacts + inference code for SageMaker deployment.

    SageMaker expects a model.tar.gz that optionally contains a 'code/'
    sub-directory with the entry-point script. We bundle everything into
    a single archive so no separate code upload is needed.

    Returns:
        Path to model.tar.gz
    """
    print("Packaging XGBoost model artifacts...")

    os.makedirs(output_dir, exist_ok=True)

    model_archive = Path(output_dir) / "model.tar.gz"
    with tarfile.open(model_archive, "w:gz") as tar:
        tar.add(model_path,        arcname="xgb_classifier.onnx")
        tar.add(label_map_path,    arcname="xgb_label_map.json")
        tar.add(inference_script,  arcname="code/inference.py")

    print(f"Model packaged: {model_archive}")
    print(f"  - xgb_classifier.onnx")
    print(f"  - xgb_label_map.json")
    print(f"  - code/inference.py")

    return str(model_archive)


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

def upload_to_s3(local_path: str, bucket: str, key_prefix: str) -> str:
    """Upload a local file to S3 and return the s3:// URI."""
    s3 = boto3.client("s3")
    filename = Path(local_path).name
    s3_key = f"{key_prefix}/{filename}"

    print(f"Uploading {filename} → s3://{bucket}/{s3_key} ...")
    s3.upload_file(local_path, bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"  Uploaded: {s3_uri}")
    return s3_uri


# ---------------------------------------------------------------------------
# SageMaker deploy (pure boto3)
# ---------------------------------------------------------------------------

def deploy_to_sagemaker(
    model_data: str,
    endpoint_name: str,
    instance_type: str,
    role: str,
    region: str,
) -> str:
    """
    Create a SageMaker model, endpoint config, and endpoint using boto3.

    Args:
        model_data:     s3:// URI of model.tar.gz
        endpoint_name:  Name for the SageMaker endpoint
        instance_type:  e.g. 'ml.t2.medium'
        role:           SageMaker execution role ARN
        region:         AWS region name

    Returns:
        Endpoint name
    """
    sm = boto3.client("sagemaker", region_name=region)

    model_name       = f"{endpoint_name}-model-{int(time.time())}"
    config_name      = f"{endpoint_name}-config-{int(time.time())}"
    image_uri        = get_image_uri(region)

    # ── 1. Create Model ──────────────────────────────────────────────────────
    print(f"\nCreating SageMaker model '{model_name}' ...")
    print(f"  Container image : {image_uri}")
    print(f"  Model data      : {model_data}")

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        ExecutionRoleArn=role,
    )
    print("  ✓ Model created")

    # ── 2. Create Endpoint Config ─────────────────────────────────────────────
    print(f"\nCreating endpoint config '{config_name}' ...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    print("  ✓ Endpoint config created")

    # ── 3. Create / Update Endpoint ──────────────────────────────────────────
    existing = [e["EndpointName"] for e in sm.list_endpoints()["Endpoints"]]
    if endpoint_name in existing:
        print(f"\nEndpoint '{endpoint_name}' already exists — updating ...")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    else:
        print(f"\nCreating endpoint '{endpoint_name}' ... (this may take 5–10 min)")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

    # ── 4. Wait for InService ───────────────────────────────────────────────
    print("Waiting for endpoint to become InService ...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60})

    print(f"\n✓ Endpoint '{endpoint_name}' is InService!")
    return endpoint_name


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_endpoint(endpoint_name: str, test_encoded: list):
    """Quick smoke-test the deployed endpoint."""
    print(f"\nTesting endpoint '{endpoint_name}' ...")

    runtime = boto3.client("sagemaker-runtime")
    payload = json.dumps({"encoded": test_encoded})

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode())

    print("✓ Endpoint test successful!")
    print(f"  Predicted label : {result['labels'][0]}")
    print(f"  Confidence      : {result['confidences'][0]:.4f}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deploy XGBoost to SageMaker (boto3 only)")

    parser.add_argument("--model-path",      type=str, required=True,  help="Path to xgb_classifier.onnx")
    parser.add_argument("--label-map-path",  type=str, required=True,  help="Path to xgb_label_map.json")
    parser.add_argument("--endpoint-name",   type=str, default="nids-xgboost",  help="SageMaker endpoint name")
    parser.add_argument("--instance-type",   type=str, default="ml.m5.large",  help="SageMaker instance type")
    parser.add_argument("--role",            type=str, required=True,  help="SageMaker execution role ARN")
    parser.add_argument("--bucket",          type=str, required=True,  help="S3 bucket name")
    parser.add_argument("--region",          type=str, default=None,   help="AWS region (default: from env/config)")
    parser.add_argument("--output-dir",      type=str, default="./deploy_output_xgb", help="Directory for local artifacts")
    parser.add_argument("--skip-test",       action="store_true",      help="Skip endpoint smoke test")

    args = parser.parse_args()

    # Resolve region
    region = args.region or boto3.session.Session().region_name or "us-east-1"

    # Resolve inference script path
    script_dir      = Path(__file__).parent
    inference_script = str(script_dir / "inference_xgboost.py")

    print("=" * 80)
    print("XGBOOST DEPLOYMENT TO SAGEMAKER")
    print("=" * 80)
    print(f"Model        : {args.model_path}")
    print(f"Label Map    : {args.label_map_path}")
    print(f"Endpoint     : {args.endpoint_name}")
    print(f"Instance     : {args.instance_type}")
    print(f"Role         : {args.role}")
    print(f"Bucket       : {args.bucket}")
    print(f"Region       : {region}")
    print("=" * 80)

    # Step 1: Package
    model_archive = package_model(
        model_path=args.model_path,
        label_map_path=args.label_map_path,
        inference_script=inference_script,
        output_dir=args.output_dir,
    )

    # Step 2: Upload to S3
    model_data = upload_to_s3(model_archive, args.bucket, "models/xgboost")

    # Step 3: Deploy
    endpoint_name = deploy_to_sagemaker(
        model_data=model_data,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        role=args.role,
        region=region,
    )

    # Step 4: Smoke test
    if not args.skip_test:
        test_encoded = [0.1] * 8  # 8-dim latent from autoencoder
        test_endpoint(endpoint_name, test_encoded)

    # Save deployment info
    deployment_info = {
        "endpoint_name": endpoint_name,
        "instance_type": args.instance_type,
        "model_data": model_data,
        "deployed_at": datetime.now().isoformat(),
        "role": args.role,
        "bucket": args.bucket,
        "region": region,
    }

    info_path = Path(args.output_dir) / "deployment_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(deployment_info, f, indent=2)

    print(f"\n✓ Deployment info saved to: {info_path}")
    print("\nNext steps:")
    print(f"  1. Update NIDS Rust config with endpoint: {endpoint_name}")
    print(f"  2. Test with: python test_endpoints.py --test-type xgboost --xgboost-endpoint {endpoint_name}")


if __name__ == "__main__":
    main()
