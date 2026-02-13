"""
Deploy XGBoost classifier to AWS SageMaker.

This script:
1. Packages the ONNX model with custom inference code
2. Uploads to S3
3. Creates SageMaker model
4. Deploys to endpoint
"""

import argparse
import json
import os
import tarfile
import boto3
import sagemaker
from sagemaker.model import Model
from pathlib import Path
from datetime import datetime


def package_model(model_path: str, label_map_path: str, output_dir: str) -> str:
    """
    Package model artifacts for SageMaker deployment.
    
    Args:
        model_path: Path to xgb_classifier.onnx
        label_map_path: Path to xgb_label_map.json
        output_dir: Directory to save packaged model
        
    Returns:
        Path to model.tar.gz
    """
    print("Packaging XGBoost model artifacts...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for packaging
    temp_dir = Path(output_dir) / "temp_model"
    temp_dir.mkdir(exist_ok=True)
    
    # Copy model and label map
    import shutil
    shutil.copy(model_path, temp_dir / "xgb_classifier.onnx")
    shutil.copy(label_map_path, temp_dir / "xgb_label_map.json")
    
    # Create tar.gz archive
    model_archive = Path(output_dir) / "model.tar.gz"
    with tarfile.open(model_archive, "w:gz") as tar:
        tar.add(temp_dir / "xgb_classifier.onnx", arcname="xgb_classifier.onnx")
        tar.add(temp_dir / "xgb_label_map.json", arcname="xgb_label_map.json")
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    
    print(f"Model packaged: {model_archive}")
    print(f"  - xgb_classifier.onnx")
    print(f"  - xgb_label_map.json")
    
    return str(model_archive)


def deploy_to_sagemaker(
    model_archive: str,
    endpoint_name: str,
    instance_type: str,
    role: str,
    bucket: str
) -> str:
    """
    Deploy XGBoost model to SageMaker endpoint.
    
    Args:
        model_archive: Path to model.tar.gz
        endpoint_name: Name for the endpoint
        instance_type: EC2 instance type
        role: SageMaker execution role ARN
        bucket: S3 bucket for model artifacts
        
    Returns:
        Endpoint name
    """
    print(f"\nDeploying to SageMaker...")
    print(f"  Endpoint: {endpoint_name}")
    print(f"  Instance: {instance_type}")
    
    # Initialize SageMaker session
    sess = sagemaker.Session(default_bucket=bucket)
    
    # Upload model to S3
    print("\nUploading model to S3...")
    model_data = sess.upload_data(
        path=model_archive,
        bucket=bucket,
        key_prefix="models/xgboost"
    )
    print(f"Model uploaded to: {model_data}")
    
    # Get inference script path
    script_dir = Path(__file__).parent
    inference_script = script_dir / "inference_xgboost.py"
    
    # Get ONNX Runtime container image
    region = sess.boto_region_name
    account_id = sess.account_id()
    
    # Use PyTorch container with ONNX Runtime
    # (SageMaker doesn't have a dedicated ONNX container, so we use PyTorch)
    image_uri = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.0.0",
        py_version="py310",
        instance_type=instance_type,
        image_scope="inference"
    )
    
    print(f"\nUsing container image: {image_uri}")
    
    # Create SageMaker model
    print("\nCreating SageMaker model...")
    
    # Package inference script with model
    temp_code_dir = Path(output_dir) / "code"
    temp_code_dir.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(inference_script, temp_code_dir / "inference.py")
    
    # Create code tarball
    code_archive = Path(output_dir) / "code.tar.gz"
    with tarfile.open(code_archive, "w:gz") as tar:
        tar.add(temp_code_dir / "inference.py", arcname="code/inference.py")
    
    shutil.rmtree(temp_code_dir)
    
    # Upload code to S3
    code_data = sess.upload_data(
        path=str(code_archive),
        bucket=bucket,
        key_prefix="models/xgboost/code"
    )
    
    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=role,
        sagemaker_session=sess,
        entry_point="inference.py",
        source_dir=None,  # Code is in model_data
        dependencies=["onnxruntime"]
    )
    
    # Deploy to endpoint
    print(f"\nDeploying endpoint '{endpoint_name}'...")
    print("This may take 5-10 minutes...")
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    print(f"\n✓ Endpoint deployed successfully!")
    print(f"  Endpoint name: {endpoint_name}")
    
    return endpoint_name


def test_endpoint(endpoint_name: str, test_encoded: list):
    """
    Test the deployed endpoint with sample data.
    
    Args:
        endpoint_name: Name of the endpoint
        test_encoded: List of encoded features (from autoencoder)
    """
    print(f"\nTesting endpoint '{endpoint_name}'...")
    
    runtime = boto3.client('sagemaker-runtime')
    
    payload = json.dumps({"encoded": test_encoded})
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    
    result = json.loads(response['Body'].read().decode())
    
    print("✓ Endpoint test successful!")
    print(f"  Predicted label: {result['labels'][0]}")
    print(f"  Confidence: {result['confidences'][0]:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy XGBoost to SageMaker")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to xgb_classifier.onnx"
    )
    parser.add_argument(
        "--label-map-path",
        type=str,
        required=True,
        help="Path to xgb_label_map.json"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="nids-xgboost",
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.t2.medium",
        help="SageMaker instance type"
    )
    parser.add_argument(
        "--role",
        type=str,
        help="SageMaker execution role ARN (if not provided, uses default)"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="S3 bucket name (if not provided, uses default)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./deploy_output_xgb",
        help="Directory for deployment artifacts"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip endpoint testing"
    )
    
    args = parser.parse_args()
    
    # Get SageMaker role
    if args.role:
        role = args.role
    else:
        role = sagemaker.get_execution_role()
    
    # Get S3 bucket
    bucket = args.bucket if args.bucket else sagemaker.Session().default_bucket()
    
    print("=" * 80)
    print("XGBOOST DEPLOYMENT TO SAGEMAKER")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Label Map: {args.label_map_path}")
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Role: {role}")
    print(f"Bucket: {bucket}")
    print("=" * 80)
    
    # Step 1: Package model
    model_archive = package_model(
        model_path=args.model_path,
        label_map_path=args.label_map_path,
        output_dir=args.output_dir
    )
    
    # Step 2: Deploy to SageMaker
    endpoint_name = deploy_to_sagemaker(
        model_archive=model_archive,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        role=role,
        bucket=bucket
    )
    
    # Step 3: Test endpoint
    if not args.skip_test:
        # Create test encoded features (8-dim from autoencoder)
        test_encoded = [0.1] * 8
        test_endpoint(endpoint_name, test_encoded)
    
    # Save deployment info
    deployment_info = {
        "endpoint_name": endpoint_name,
        "instance_type": args.instance_type,
        "model_archive": model_archive,
        "deployed_at": datetime.now().isoformat(),
        "role": role,
        "bucket": bucket
    }
    
    info_path = Path(args.output_dir) / "deployment_info.json"
    with open(info_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\n✓ Deployment info saved to: {info_path}")
    print("\nNext steps:")
    print(f"  1. Update NIDS config with endpoint: {endpoint_name}")
    print(f"  2. Test end-to-end pipeline (autoencoder → XGBoost)")
    print(f"  3. Deploy NIDS server with SageMaker integration")


if __name__ == "__main__":
    main()
