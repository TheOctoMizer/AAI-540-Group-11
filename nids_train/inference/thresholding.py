# inference/thresholding.py
import numpy as np

def compute_thresholds(errors):
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "p999": float(np.percentile(errors, 99.9)),
    }