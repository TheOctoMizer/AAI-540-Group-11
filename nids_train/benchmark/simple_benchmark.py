"""
Simple latency benchmark helper for NIDS models.
Provides the exact benchmark function as requested.
"""

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

def benchmark(session, x, runs=1000):
    """
    Benchmark helper function for ONNX inference latency.
    
    Args:
        session: ONNX Runtime inference session
        x: Input numpy array
        runs: Number of benchmark runs (default: 1000)
    
    Returns:
        Average latency in milliseconds
    """
    # Warm-up
    for _ in range(50):
        session.run(None, {"input": x})

    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"input": x})
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / runs
    return avg_ms

def run_simple_benchmark():
    """Run simple benchmark comparing FP32 vs INT8 models."""
    print("Simple NIDS Latency Benchmark")
    print("=" * 40)
    
    # Configuration
    model_dir = Path("../model")
    input_dim = 77
    
    # Create simple test input
    x_np = np.random.randn(1, input_dim).astype(np.float32)
    print(f"Test input shape: {x_np.shape}")
    
    # Load ONNX sessions
    try:
        sess_fp32 = ort.InferenceSession(
            str(model_dir / "autoencoder.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print("Loaded ONNX FP32 model")
    except Exception as e:
        print(f"Failed to load FP32 model: {e}")
        return
    
    try:
        sess_int8 = ort.InferenceSession(
            str(model_dir / "autoencoder_int8.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print("Loaded ONNX INT8 model")
    except Exception as e:
        print(f"Failed to load INT8 model: {e}")
        return
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    fp32_ms = benchmark(sess_fp32, x_np)
    int8_ms = benchmark(sess_int8, x_np)
    
    # Print results
    print(f"\nResults:")
    print(f"ONNX FP32 avg latency: {fp32_ms:.4f} ms")
    print(f"ONNX INT8 avg latency: {int8_ms:.4f} ms")
    print(f"Speedup: {fp32_ms / int8_ms:.2f}×")
    
    # Additional metrics
    fp32_throughput = 1000.0 / fp32_ms
    int8_throughput = 1000.0 / int8_ms
    
    print(f"\nThroughput:")
    print(f"FP32: {fp32_throughput:.0f} inferences/second")
    print(f"INT8: {int8_throughput:.0f} inferences/second")

if __name__ == "__main__":
    run_simple_benchmark()
