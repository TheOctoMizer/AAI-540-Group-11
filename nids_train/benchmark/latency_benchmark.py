"""
Latency benchmark for NIDS models (Realistic NIDS Style).

This script benchmarks the inference latency of:
1. PyTorch model (FP32)
2. ONNX FP32 model 
3. ONNX INT8 quantized model

Provides realistic NIDS-style performance metrics with warm-up runs
and statistical analysis.
"""

import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
import json
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder

def load_scaler_params(scaler_path):
    """Load scaler parameters from JSON file."""
    with open(scaler_path, 'r') as f:
        return json.load(f)

def create_realistic_input(scaler_params, input_dim=77):
    """
    Create realistic network traffic input data for benchmarking.
    Uses typical ranges for network flow features.
    """
    # Create realistic feature values based on typical network traffic
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

def benchmark_pytorch(model, x_torch, runs=1000):
    """Benchmark PyTorch model inference latency."""
    # Warm-up
    for _ in range(50):
        with torch.no_grad():
            _ = model(x_torch)
    
    # Synchronize CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(x_torch)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_ms = (end - start) * 1000 / runs
    return avg_ms

def benchmark_onnx(session, x_np, runs=1000):
    """Benchmark ONNX model inference latency."""
    # Warm-up
    for _ in range(50):
        _ = session.run(None, {"input": x_np})
    
    start = time.perf_counter()
    for _ in range(runs):
        _ = session.run(None, {"input": x_np})
    end = time.perf_counter()
    
    avg_ms = (end - start) * 1000 / runs
    return avg_ms

def run_multiple_benchmarks(model, sess_fp32, sess_int8, x_np, x_torch, num_iterations=5):
    """Run multiple benchmark iterations and provide statistics."""
    print("Running multiple benchmark iterations for statistical accuracy...")
    
    pytorch_times = []
    onnx_fp32_times = []
    onnx_int8_times = []
    
    for i in range(num_iterations):
        print(f"  Iteration {i+1}/{num_iterations}...")
        
        pytorch_ms = benchmark_pytorch(model, x_torch, runs=1000)
        fp32_ms = benchmark_onnx(sess_fp32, x_np, runs=1000)
        int8_ms = benchmark_onnx(sess_int8, x_np, runs=1000)
        
        pytorch_times.append(pytorch_ms)
        onnx_fp32_times.append(fp32_ms)
        onnx_int8_times.append(int8_ms)
    
    return pytorch_times, onnx_fp32_times, onnx_int8_times

def print_statistics(name, times):
    """Print statistical summary of benchmark times."""
    times_array = np.array(times)
    print(f"\n{name}:")
    print(f"  Mean:   {times_array.mean():.4f} ms")
    print(f"  Std:    {times_array.std():.4f} ms")
    print(f"  Min:    {times_array.min():.4f} ms")
    print(f"  Max:    {times_array.max():.4f} ms")
    print(f"  Median: {np.median(times_array):.4f} ms")

def main():
    """Main benchmark function."""
    print("=" * 80)
    print("LATENCY BENCHMARK: NIDS Models (Realistic NIDS Style)")
    print("=" * 80)
    
    # Configuration
    input_dim = 77
    model_dir = Path("../model")
    scaler_path = Path("../scaler_params.json")
    
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
        print("\nPlease run training and export first to generate model files.")
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
    
    # Load models
    print("\nLoading models...")
    
    # PyTorch model
    model = Autoencoder(input_dim=input_dim, latent_dim=8)
    model.load_state_dict(torch.load(model_dir / "autoencoder.pt", map_location="cpu"))
    model.eval()
    
    # ONNX FP32 model
    sess_fp32 = ort.InferenceSession(
        str(model_dir / "autoencoder.onnx"),
        providers=["CPUExecutionProvider"]
    )
    
    # ONNX INT8 model
    sess_int8 = ort.InferenceSession(
        str(model_dir / "autoencoder_int8.onnx"),
        providers=["CPUExecutionProvider"]
    )
    
    print("All models loaded successfully!")
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("RUNNING LATENCY BENCHMARKS")
    print("=" * 80)
    
    pytorch_times, onnx_fp32_times, onnx_int8_times = run_multiple_benchmarks(
        model, sess_fp32, sess_int8, x_np, x_torch, num_iterations=5
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    print_statistics("PyTorch FP32", pytorch_times)
    print_statistics("ONNX FP32", onnx_fp32_times)
    print_statistics("ONNX INT8", onnx_int8_times)
    
    # Calculate speedups
    pytorch_mean = np.mean(pytorch_times)
    fp32_mean = np.mean(onnx_fp32_times)
    int8_mean = np.mean(onnx_int8_times)
    
    print(f"\n" + "=" * 80)
    print("SPEEDUP ANALYSIS")
    print("=" * 80)
    
    print(f"PyTorch vs ONNX FP32 speedup: {pytorch_mean / fp32_mean:.2f}×")
    print(f"ONNX FP32 vs INT8 speedup:     {fp32_mean / int8_mean:.2f}×")
    print(f"PyTorch vs ONNX INT8 speedup:  {pytorch_mean / int8_mean:.2f}×")
    
    # NIDS-specific analysis
    print(f"\n" + "=" * 80)
    print("NIDS PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Calculate packets per second capability
    packets_per_second_pytorch = 1000.0 / pytorch_mean
    packets_per_second_fp32 = 1000.0 / fp32_mean
    packets_per_second_int8 = 1000.0 / int8_mean
    
    print(f"PyTorch FP32 throughput: {packets_per_second_pytorch:.0f} packets/second")
    print(f"ONNX FP32 throughput:    {packets_per_second_fp32:.0f} packets/second")
    print(f"ONNX INT8 throughput:    {packets_per_second_int8:.0f} packets/second")
    
    # Network capability assessment
    typical_1gbps = 1000000000 / 8 / 1500  # ~83K packets/sec for 1500-byte packets
    typical_10gbps = 10000000000 / 8 / 1500  # ~833K packets/sec for 1500-byte packets
    
    print(f"\nNetwork Line Rate Capability:")
    print(f"  1 Gbps line rate: {typical_1gbps:.0f} packets/sec (1500-byte packets)")
    print(f"  10 Gbps line rate: {typical_10gbps:.0f} packets/sec (1500-byte packets)")
    
    print(f"\nReal-time NIDS Assessment:")
    if packets_per_second_int8 >= typical_1gbps:
        print("  ONNX INT8 can handle 1 Gbps line rate")
    else:
        print(f"  ONNX INT8 cannot handle 1 Gbps line rate ({packets_per_second_int8/typical_1gbps*100:.1f}% capacity)")
    
    if packets_per_second_int8 >= typical_10gbps:
        print("  ONNX INT8 can handle 10 Gbps line rate")
    else:
        print(f"  ONNX INT8 cannot handle 10 Gbps line rate ({packets_per_second_int8/typical_10gbps*100:.1f}% capacity)")
    
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()
