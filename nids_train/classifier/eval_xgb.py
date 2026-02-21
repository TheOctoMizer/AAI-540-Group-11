import numpy as np
import onnxruntime as ort
import xgboost as xgb
from pathlib import Path

from preprocessing.nids_preprocessor import NIDSPreprocessor


def evaluate_xgb_onnx_consistency(
    model_path,
    onnx_path,
    data_dir,
    dataset,
    autoencoder_path=None,
    latent_dim=8,
    num_samples=10,
):
    """
    Evaluate consistency between XGBoost and ONNX models.
    
    When autoencoder_path is provided, raw features are first encoded
    through the autoencoder (matching the training pipeline).

    Returns:
        tuple: (max_difference, mean_difference, xgb_probs, onnx_probs)
    """
    # Load scaler & data
    preprocessor = NIDSPreprocessor("scaler_params.json")
    X, labels = preprocessor.preprocess_data(
        Path(data_dir) / dataset,
        fit_scaler=False
    )

    mask = labels != "BENIGN"
    X_attack = X[mask][:num_samples]

    # Encode through autoencoder if provided
    if autoencoder_path is not None:
        from .train import _encode_features
        X_attack = _encode_features(X_attack, autoencoder_path, latent_dim=latent_dim)

    # Load models
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    sess = ort.InferenceSession(onnx_path)

    # Compare outputs
    xgb_probs = xgb_model.predict_proba(X_attack)
    onnx_output = sess.run(None, {"input": X_attack.astype(np.float32)})
    
    # ONNX output might be probabilities or logits, handle both cases
    if len(onnx_output) == 1:
        onnx_probs = onnx_output[0]
    else:
        onnx_probs = onnx_output[1]  # Usually second output is probabilities
    
    # Ensure both arrays have the same shape
    if onnx_probs.shape != xgb_probs.shape:
        print(f"Shape mismatch: XGB {xgb_probs.shape} vs ONNX {onnx_probs.shape}")
        # Try to handle different output formats
        if len(onnx_probs.shape) == 1:
            onnx_probs = np.eye(xgb_probs.shape[1])[onnx_probs.astype(int)]
    
    max_diff = np.max(np.abs(xgb_probs - onnx_probs))
    mean_diff = np.mean(np.abs(xgb_probs - onnx_probs))

    return max_diff, mean_diff, xgb_probs, onnx_probs


def main():
    """Standalone script entry point."""
    max_diff, mean_diff, _, _ = evaluate_xgb_onnx_consistency(
        model_path=Path("model/xgb_classifier.json"),
        onnx_path=Path("model/xgb_classifier.onnx"),
        data_dir="nids_dataset",
        dataset="Wednesday-workingHours.pcap_ISCX.csv"
    )
    
    print(f"Max probability difference: {max_diff:.6e}")
    print(f"Mean probability difference: {mean_diff:.6e}")


if __name__ == "__main__":
    main()