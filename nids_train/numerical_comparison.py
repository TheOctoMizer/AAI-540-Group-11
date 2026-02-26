"""
Numerical comparison between PyTorch and ONNX models (FP32 & INT8).

This script:
1. Creates realistic test input (not random noise)
2. Compares PyTorch vs ONNX FP32 outputs
3. Compares ONNX FP32 vs ONNX INT8 outputs
4. Reports maximum absolute differences
"""

import numpy as np
import torch
import joblib
import json
import onnxruntime as ort
from pathlib import Path

# Add current directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

from models.autoencoder import Autoencoder

def load_scaler_params(scaler_path):
    """Load scaler parameters from JSON file."""
    with open(scaler_path, 'r') as f:
        return json.load(f)

def create_realistic_input(scaler_params, input_dim=77):
    """
    Create realistic network traffic input data.
    Uses typical ranges for network flow features.
    """
    # Create realistic feature values based on typical network traffic
    # These are approximate ranges for common network flow features
    features = []
    
    # Duration (0-300 seconds)
    features.append(np.random.uniform(0.1, 50.0))
    
    # Total packets (forward + backward)
    total_packets = np.random.randint(10, 10000)
    features.append(total_packets)
    
    # Total bytes (forward + backward) 
    total_bytes = np.random.randint(1000, 10000000)
    features.append(total_bytes)
    
    # Packet length statistics
    avg_pkt_len = total_bytes / max(total_packets, 1)
    features.append(avg_pkt_len)  # avg packet length
    features.append(np.random.uniform(40, 1500))  # max packet length
    features.append(np.random.uniform(40, avg_pkt_len * 2))  # min packet length
    
    # Inter-arrival times
    features.append(np.random.uniform(0.001, 1.0))  # mean flow interval
    features.append(np.random.uniform(0.0001, 0.1))  # std flow interval
    features.append(np.random.uniform(0.0001, 0.5))  # max flow interval
    features.append(np.random.uniform(0.00001, 0.01))  # min flow interval
    
    # TCP flags (binary/categorical features)
    features.extend([
        np.random.randint(0, 2),  # FIN flag
        np.random.randint(0, 2),  # SYN flag
        np.random.randint(0, 2),  # RST flag
        np.random.randint(0, 2),  # PSH flag
        np.random.randint(0, 2),  # ACK flag
        np.random.randint(0, 2),  # URG flag
        np.random.randint(0, 2),  # ECE flag
        np.random.randint(0, 2),  # CWR flag
    ])
    
    # Down/Up ratio
    features.append(np.random.uniform(0.1, 10.0))
    
    # Average packet size (forward/backward)
    features.extend([
        np.random.uniform(40, 1500),
        np.random.uniform(40, 1500),
    ])
    
    # Flow bytes/s and packets/s
    features.extend([
        total_bytes / max(np.random.uniform(1, 300), 0.001),  # Flow bytes/s
        total_packets / max(np.random.uniform(1, 300), 0.001),  # Flow packets/s
    ])
    
    # Fill remaining features with realistic values
    remaining_features = input_dim - len(features)
    if remaining_features > 0:
        # Add statistical features (means, stds, max, mins)
        for i in range(remaining_features // 4):
            base_val = np.random.uniform(1, 1000)
            features.extend([
                base_val * np.random.uniform(0.8, 1.2),  # mean
                base_val * np.random.uniform(0.1, 0.5),  # std
                base_val * np.random.uniform(1.5, 3.0),  # max
                base_val * np.random.uniform(0.1, 0.8),  # min
            ])
        
        # Add any remaining features
        while len(features) < input_dim:
            features.append(np.random.uniform(0.001, 100))
    
    # Trim to exact input_dim
    features = features[:input_dim]
    
    # Convert to numpy array and scale
    x_np = np.array(features, dtype=np.float32).reshape(1, -1)
    
    # Apply scaling if scaler params are available
    if scaler_params and 'scale' in scaler_params:
        scale_array = np.array(scaler_params['scale'], dtype=np.float32)
        if len(scale_array) == input_dim:
            x_np = x_np * scale_array
    
    return x_np

def max_abs_diff(a, b):
    """Calculate maximum absolute difference between two arrays."""
    return float(np.max(np.abs(a - b)))

def main():
    """Main comparison function."""
    print("=" * 80)
    print("NUMERICAL COMPARISON: PyTorch vs ONNX (FP32 & INT8)")
    print("=" * 80)
    
    # Configuration
    input_dim = 62
    model_dir = Path("model")
    scaler_path = Path("scaler_params.json")
    
    # Check if model files exist
    required_files = [
        model_dir / "autoencoder.pt",
        model_dir / "autoencoder.onnx", 
        model_dir / "autoencoder_int8.onnx"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("Missing model files:")
        for f in missing_files:
            print(f"   {f}")
        print("\nPlease run training first to generate model files.")
        return
    
    # Load scaler parameters
    scaler_params = None
    if scaler_path.exists():
        print(f"Loading scaler parameters from {scaler_path}")
        scaler_params = load_scaler_params(scaler_path)
    else:
        print(f"Scaler file not found: {scaler_path}")
    
    # Create realistic test input
    print(f"Creating realistic test input (dim={input_dim})")
    x_np = create_realistic_input(scaler_params, input_dim)
    x_torch = torch.from_numpy(x_np)
    
    print(f"Input shape: {x_np.shape}")
    print(f"Input range: [{x_np.min():.6f}, {x_np.max():.6f}]")
    
    # PyTorch inference
    print("\nRunning PyTorch inference...")
    model = Autoencoder(input_dim=input_dim, latent_dim=8)
    model.load_state_dict(torch.load(model_dir / "autoencoder.pt", map_location="cpu"))
    model.eval()
    
    with torch.no_grad():
        out_torch = model(x_torch).numpy()
    
    print(f"PyTorch output shape: {out_torch.shape}")
    print(f"PyTorch output range: [{out_torch.min():.6f}, {out_torch.max():.6f}]")
    
    # ONNX FP32 inference
    print("\nRunning ONNX FP32 inference...")
    sess_fp32 = ort.InferenceSession(
        str(model_dir / "autoencoder.onnx"),
        providers=["CPUExecutionProvider"]
    )
    
    out_onnx_fp32 = sess_fp32.run(None, {"input": x_np})[0]
    
    print(f"ONNX FP32 output shape: {out_onnx_fp32.shape}")
    print(f"ONNX FP32 output range: [{out_onnx_fp32.min():.6f}, {out_onnx_fp32.max():.6f}]")
    
    # ONNX INT8 inference
    print("\nRunning ONNX INT8 inference...")
    sess_int8 = ort.InferenceSession(
        str(model_dir / "autoencoder_int8.onnx"),
        providers=["CPUExecutionProvider"]
    )
    
    out_onnx_int8 = sess_int8.run(None, {"input": x_np})[0]
    
    print(f"ONNX INT8 output shape: {out_onnx_int8.shape}")
    print(f"ONNX INT8 output range: [{out_onnx_int8.min():.6f}, {out_onnx_int8.max():.6f}]")
    
    # Numerical comparisons
    print("\n" + "=" * 80)
    print("NUMERICAL DIFFERENCES")
    print("=" * 80)
    
    diff_torch_fp32 = max_abs_diff(out_torch, out_onnx_fp32)
    diff_fp32_int8 = max_abs_diff(out_onnx_fp32, out_onnx_int8)
    
    print(f"Torch vs ONNX FP32 diff: {diff_torch_fp32:.8f}")
    print(f"FP32 vs INT8 diff:       {diff_fp32_int8:.8f}")
    
    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"PyTorch mean:   {out_torch.mean():.6f}")
    print(f"ONNX FP32 mean: {out_onnx_fp32.mean():.6f}")
    print(f"ONNX INT8 mean: {out_onnx_int8.mean():.6f}")
    
    print(f"\nPyTorch std:    {out_torch.std():.6f}")
    print(f"ONNX FP32 std:  {out_onnx_fp32.std():.6f}")
    print(f"ONNX INT8 std:  {out_onnx_int8.std():.6f}")
    
    # Quality assessment
    print(f"\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    
    if diff_torch_fp32 < 1e-5:
        print("PyTorch vs ONNX FP32: EXCELLENT match (< 1e-5)")
    elif diff_torch_fp32 < 1e-4:
        print("PyTorch vs ONNX FP32: GOOD match (< 1e-4)")
    elif diff_torch_fp32 < 1e-3:
        print("PyTorch vs ONNX FP32: ACCEPTABLE match (< 1e-3)")
    else:
        print("PyTorch vs ONNX FP32: POOR match (> 1e-3)")
    
    if diff_fp32_int8 < 1e-2:
        print("FP32 vs INT8: EXCELLENT quantization (< 1e-2)")
    elif diff_fp32_int8 < 1e-1:
        print("FP32 vs INT8: GOOD quantization (< 1e-1)")
    elif diff_fp32_int8 < 1.0:
        print("FP32 vs INT8: ACCEPTABLE quantization (< 1.0)")
    else:
        print("FP32 vs INT8: POOR quantization (> 1.0)")
    
    print("\nComparison completed successfully!")

if __name__ == "__main__":
    main()
