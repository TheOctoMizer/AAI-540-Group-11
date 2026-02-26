import json
import numpy as np
import pandas as pd
import onnxruntime as ort

def get_mse(data, sess):
    inputs = {sess.get_inputs()[0].name: data.astype(np.float32)}
    outputs = sess.run(None, inputs)
    recon = outputs[0]  # First output is reconstruction
    return np.mean((data - recon)**2, axis=1)

print("Loading ONNX model...")
sess = ort.InferenceSession("nids_train/model/autoencoder.onnx")

print("Loading val (benign) data...")
df_benign = pd.read_parquet("nids_train/output/features/val/features.parquet")
mse_benign = get_mse(df_benign.values, sess)

print("\nMSE Percentiles for Benign (val):")
print(f"  Median: {np.median(mse_benign):.6f}")
print(f"  P90:    {np.percentile(mse_benign, 90):.6f}")
print(f"  P95:    {np.percentile(mse_benign, 95):.6f}")
print(f"  P99:    {np.percentile(mse_benign, 99):.6f}")

print("\nLoading test (mixed) data...")
df_test = pd.read_parquet("nids_train/output/features/test/features.parquet")
labels_test = pd.read_csv("nids_train/output/features/test/labels.csv")["label"].values

print("\nMedian MSE by Attack Type (test):")
attack_types = np.unique(labels_test)
for atype in attack_types:
    mask = labels_test == atype
    data = df_test.values[mask]
    mse = get_mse(data, sess)
    print(f"  {atype}: median={np.median(mse):.6f}, min={np.min(mse):.6f} max={np.max(mse):.6f} (N={len(mse)})")
