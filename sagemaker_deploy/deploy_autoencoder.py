"""
Deploy Autoencoder model to AWS SageMaker.

This script:
1. Packages the PyTorch model with custom inference code
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
from sagemaker.pytorch import PyTorchModel
from pathlib import Path
from datetime import datetime


def package_model(model_path: str, scaler_path: str, output_dir: str) -> str:
    """
    Package model artifacts for SageMaker deployment.
    
    Args:
        model_path: Path to autoencoder.pt
        scaler_path: Path to scaler_params.json
        output_dir: Directory to save packaged model
        
    Returns:
        Path to model.tar.gz
    """
    print("Packaging model artifacts...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scaler params to get input_dim
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    input_dim = scaler_params['n_features']
    
    # Create model metadata
    metadata = {
        "input_dim": input_dim,
        "latent_dim": 8,
        "created_at": datetime.now().isoformat(),
        "framework": "pytorch",
        "framework_version": "2.0.0"
    }
    
    # Create temporary directory for packaging
    temp_dir = Path(output_dir) / "temp_model"
    temp_dir.mkdir(exist_ok=True)
    
    # Copy model file
    import shutil
    shutil.copy(model_path, temp_dir / "autoencoder.pt")
    
    # Save metadata
    with open(temp_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy scaler params
    shutil.copy(scaler_path, temp_dir / "scaler_params.json")
    
    # Create tar.gz archive
    model_archive = Path(output_dir) / "model.tar.gz"
    with tarfile.open(model_archive, "w:gz") as tar:
        tar.add(temp_dir / "autoencoder.pt", arcname="autoencoder.pt")
        tar.add(temp_dir / "model_metadata.json", arcname="model_metadata.json")
        tar.add(temp_dir / "scaler_params.json", arcname="scaler_params.json")
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    
    print(f"Model packaged: {model_archive}")
    print(f"  - autoencoder.pt")
    print(f"  - model_metadata.json (input_dim={input_dim})")
    print(f"  - scaler_params.json")
    
    return str(model_archive)


def deploy_to_sagemaker(
    model_archive: str,
    endpoint_name: str,
    instance_type: str,
    role: str,
    bucket: str
) -> str:
    """
    Deploy model to SageMaker endpoint.
    
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
        key_prefix="models/autoencoder"
    )
    print(f"Model uploaded to: {model_data}")
    
    # Get inference script path
    script_dir = Path(__file__).parent
    inference_script = script_dir / "inference_autoencoder.py"
    
    # Create PyTorch model
    print("\nCreating SageMaker model...")
    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role,
        entry_point=str(inference_script),
        framework_version="2.0.0",
        py_version="py310",
        sagemaker_session=sess
    )
    
    # Deploy to endpoint
    print(f"\nDeploying endpoint '{endpoint_name}'...")
    print("This may take 5-10 minutes...")
    
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    print(f"\n✓ Endpoint deployed successfully!")
    print(f"  Endpoint name: {endpoint_name}")
    print(f"  Endpoint ARN: {predictor.endpoint}")
    
    return endpoint_name


def test_endpoint(endpoint_name: str, test_features: list):
    """
    Test the deployed endpoint with sample data.
    
    Args:
        endpoint_name: Name of the endpoint
        test_features: List of 77 features
    """
    print(f"\nTesting endpoint '{endpoint_name}'...")
    
    runtime = boto3.client('sagemaker-runtime')
    
    payload = json.dumps({"features": test_features})
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )
    
    result = json.loads(response['Body'].read().decode())
    
    print("✓ Endpoint test successful!")
    print(f"  MSE Error: {result['mse_error'][0]:.6f}")
    print(f"  Encoded dim: {len(result['encoded'][0])}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy Autoencoder to SageMaker")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to autoencoder.pt"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        required=True,
        help="Path to scaler_params.json"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="nids-autoencoder",
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
        default="./deploy_output",
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
    print("AUTOENCODER DEPLOYMENT TO SAGEMAKER")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Instance: {args.instance_type}")
    print(f"Role: {role}")
    print(f"Bucket: {bucket}")
    print("=" * 80)
    
    # Step 1: Package model
    model_archive = package_model(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
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
        # Create test features (77 zeros as dummy)
        test_features = [0.1] * 77
        test_endpoint(endpoint_name, test_features)
    
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
    print(f"  2. Deploy XGBoost classifier")
    print(f"  3. Test end-to-end pipeline")


if __name__ == "__main__":
    main()
