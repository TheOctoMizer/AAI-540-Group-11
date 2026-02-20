"""
Test deployed SageMaker endpoints.

This script tests both autoencoder and XGBoost endpoints
to verify they're working correctly.
"""

import argparse
import json
import boto3
import numpy as np
from pathlib import Path


def test_autoencoder(endpoint_name: str, num_features: int = 77):
    """Test autoencoder endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing Autoencoder Endpoint: {endpoint_name}")
    print(f"{'='*60}")
    
    runtime = boto3.client('sagemaker-runtime')
    
    # Create test features
    test_features = np.random.rand(num_features).tolist()
    
    payload = json.dumps({"features": test_features})
    
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        result = json.loads(response['Body'].read().decode())
        
        print("✓ Autoencoder endpoint test PASSED")
        print(f"  MSE Error: {result['mse_error'][0]:.6f}")
        print(f"  Encoded dim: {len(result['encoded'][0])}")
        print(f"  Reconstruction dim: {len(result['reconstruction'][0])}")
        
        return result
        
    except Exception as e:
        print(f"✗ Autoencoder endpoint test FAILED")
        print(f"  Error: {str(e)}")
        return None


def test_xgboost(endpoint_name: str, encoded_dim: int = 8):
    """Test XGBoost endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing XGBoost Endpoint: {endpoint_name}")
    print(f"{'='*60}")
    
    runtime = boto3.client('sagemaker-runtime')
    
    # Create test encoded features
    test_encoded = np.random.rand(encoded_dim).tolist()
    
    payload = json.dumps({"encoded": test_encoded})
    
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        result = json.loads(response['Body'].read().decode())
        
        print("✓ XGBoost endpoint test PASSED")
        print(f"  Predicted label: {result['labels'][0]}")
        print(f"  Confidence: {result['confidences'][0]:.4f}")
        print(f"  Class ID: {result['class_ids'][0]}")
        
        return result
        
    except Exception as e:
        print(f"✗ XGBoost endpoint test FAILED")
        print(f"  Error: {str(e)}")
        return None


def test_end_to_end(autoencoder_endpoint: str, xgboost_endpoint: str):
    """Test full pipeline: features → autoencoder → XGBoost."""
    print(f"\n{'='*60}")
    print("Testing End-to-End Pipeline")
    print(f"{'='*60}")
    
    runtime = boto3.client('sagemaker-runtime')
    
    # Step 1: Generate test features
    test_features = np.random.rand(77).tolist()
    print(f"Step 1: Generated {len(test_features)} test features")
    
    # Step 2: Call autoencoder
    print(f"Step 2: Calling autoencoder endpoint...")
    ae_payload = json.dumps({"features": test_features})
    
    try:
        ae_response = runtime.invoke_endpoint(
            EndpointName=autoencoder_endpoint,
            ContentType='application/json',
            Body=ae_payload
        )
        
        ae_result = json.loads(ae_response['Body'].read().decode())
        mse_error = ae_result['mse_error'][0]
        encoded = ae_result['encoded'][0]
        
        print(f"  ✓ Autoencoder response received")
        print(f"    MSE Error: {mse_error:.6f}")
        
    except Exception as e:
        print(f"  ✗ Autoencoder failed: {str(e)}")
        return None
    
    # Step 3: Call XGBoost with encoded features
    print(f"Step 3: Calling XGBoost endpoint...")
    xgb_payload = json.dumps({"encoded": encoded})
    
    try:
        xgb_response = runtime.invoke_endpoint(
            EndpointName=xgboost_endpoint,
            ContentType='application/json',
            Body=xgb_payload
        )
        
        xgb_result = json.loads(xgb_response['Body'].read().decode())
        
        print(f"  ✓ XGBoost response received")
        print(f"    Predicted: {xgb_result['labels'][0]}")
        print(f"    Confidence: {xgb_result['confidences'][0]:.4f}")
        
    except Exception as e:
        print(f"  ✗ XGBoost failed: {str(e)}")
        return None
    
    # Step 4: Decision logic
    print(f"\nStep 4: Detection Decision")
    threshold = 0.05  # Example threshold
    
    if mse_error < threshold:
        decision = "BENIGN (passed through)"
        print(f"  Decision: {decision}")
        print(f"  Reason: MSE error ({mse_error:.6f}) < threshold ({threshold})")
    else:
        decision = f"ANOMALOUS - Classified as {xgb_result['labels'][0]}"
        print(f"  Decision: {decision}")
        print(f"  Reason: MSE error ({mse_error:.6f}) >= threshold ({threshold})")
    
    print(f"\n✓ End-to-end pipeline test PASSED")
    
    return {
        "mse_error": mse_error,
        "classification": xgb_result['labels'][0],
        "confidence": xgb_result['confidences'][0],
        "decision": decision
    }


def main():
    parser = argparse.ArgumentParser(description="Test SageMaker endpoints")
    
    parser.add_argument(
        "--autoencoder-endpoint",
        type=str,
        default="nids-autoencoder",
        help="Autoencoder endpoint name"
    )
    parser.add_argument(
        "--xgboost-endpoint",
        type=str,
        default="nids-xgboost",
        help="XGBoost endpoint name"
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["autoencoder", "xgboost", "end-to-end", "all"],
        default="all",
        help="Type of test to run"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAGEMAKER ENDPOINT TESTING")
    print("=" * 60)
    
    if args.test_type in ["autoencoder", "all"]:
        test_autoencoder(args.autoencoder_endpoint)
    
    if args.test_type in ["xgboost", "all"]:
        test_xgboost(args.xgboost_endpoint)
    
    if args.test_type in ["end-to-end", "all"]:
        test_end_to_end(args.autoencoder_endpoint, args.xgboost_endpoint)
    
    print(f"\n{'='*60}")
    print("Testing Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
