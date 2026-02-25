import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing.nids_preprocessor import NIDSPreprocessor
from export.export_xgb_onnx import export_xgb_to_onnx
from .eval_xgb import evaluate_xgb_onnx_consistency


# ---------------------------------------------------------------------------
# Local copy of the autoencoder architecture (must match training)
# ---------------------------------------------------------------------------
class _Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return reconstruction, encoded


def _encode_features(X: np.ndarray, autoencoder_path: str, latent_dim: int = 8) -> np.ndarray:
    """
    Encode raw features through the autoencoder's encoder.

    Args:
        X: Raw features array of shape (N, 77)
        autoencoder_path: Path to autoencoder.pt weights
        latent_dim: Size of the latent bottleneck (default 8)

    Returns:
        Encoded features of shape (N, latent_dim)
    """
    device = torch.device("cpu")
    input_dim = X.shape[1]

    model = _Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(autoencoder_path, map_location=device), strict=False)
    model.eval()

    print(f"Autoencoder loaded from {autoencoder_path}")
    print(f"  Input dim : {input_dim}")
    print(f"  Latent dim: {latent_dim}")

    with torch.no_grad():
        tensor_in = torch.FloatTensor(X).to(device)
        encoded = model.encoder(tensor_in).cpu().numpy()

    print(f"  Encoded shape: {encoded.shape}")
    return encoded.astype(np.float32)


def train_xgboost_classifier(
    data_dir,
    dataset,
    output_dir,
    autoencoder_path=None,
    latent_dim=8,
    export_onnx=True,
    evaluate_consistency=True,
):
    """
    Train XGBoost classifier for attack classification.

    When *autoencoder_path* is provided the raw features are first encoded
    through the autoencoder so that XGBoost learns on the latent space.

    Returns:
        tuple: (trained_model, label_encoder, test_accuracy, onnx_consistency)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load & preprocess (Global Attacks)
    # -----------------------------
    preprocessor = NIDSPreprocessor("scaler_params.json")
    
    print(f"Aggregating all attack samples from {data_dir} ...")
    X_attack, y_attack = preprocessor.aggregate_attack_data(data_dir)

    print(f"Total attack samples: {len(X_attack)}")
    unique_classes, counts = np.unique(y_attack, return_counts=True)
    print("Attack classes distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  {cls}: {count}")

    # -----------------------------
    # Encode through autoencoder
    # -----------------------------
    if autoencoder_path is not None:
        print("\nEncoding attack features through autoencoder ...")
        X_attack = _encode_features(X_attack, autoencoder_path, latent_dim=latent_dim)
    else:
        print("\n[WARN] No autoencoder provided — training on raw features.")

    input_dim = X_attack.shape[1]
    print(f"XGBoost input dim: {input_dim}")

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_attack)

    # Save label mapping
    label_map = {int(i): cls for i, cls in enumerate(encoder.classes_)}
    with open(output_dir / "xgb_label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_attack,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # -----------------------------
    # Train XGBoost
    # -----------------------------
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        early_stopping_rounds=20,
        n_jobs=-1,
    )

    print("Training XGBoost classifier...")
    # Compute balanced sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save model
    model_path = output_dir / "xgb_classifier.json"
    model.save_model(model_path)
    print(f"XGBoost model saved to {model_path}")

    # Export to ONNX if requested
    onnx_consistency = None
    if export_onnx:
        print("\nExporting to ONNX format...")
        onnx_path = output_dir / "xgb_classifier.onnx"
        label_map_path = output_dir / "xgb_label_map.json"

        export_xgb_to_onnx(
            model_path=model_path,
            label_map_path=label_map_path,
            output_path=onnx_path,
            input_dim=input_dim,
        )

        # Evaluate consistency if requested
        if evaluate_consistency:
            print("Evaluating XGBoost vs ONNX consistency...")
            max_diff, mean_diff, _, _ = evaluate_xgb_onnx_consistency(
                model_path=model_path,
                onnx_path=onnx_path,
                data_dir=data_dir,
                dataset=dataset,
                autoencoder_path=autoencoder_path,
                latent_dim=latent_dim,
                num_samples=100,
            )

            onnx_consistency = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
            }

            print(f"Max probability difference: {max_diff:.6e}")
            print(f"Mean probability difference: {mean_diff:.6e}")

            if max_diff < 1e-6:
                print("Excellent ONNX conversion (diff < 1e-6)")
            elif max_diff < 1e-4:
                print("Good ONNX conversion (diff < 1e-4)")
            else:
                print("Poor ONNX conversion (diff >= 1e-4)")

    return model, encoder, model.score(X_test, y_test), onnx_consistency