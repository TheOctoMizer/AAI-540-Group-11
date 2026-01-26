import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing.nids_preprocessor import NIDSPreprocessor
from export.export_xgb_onnx import export_xgb_to_onnx
from .eval_xgb import evaluate_xgb_onnx_consistency


def train_xgboost_classifier(data_dir, dataset, output_dir, export_onnx=True, evaluate_consistency=True):
    """
    Train XGBoost classifier for attack classification.
    
    Args:
        data_dir (str): Directory containing the dataset
        dataset (str): Dataset filename
        output_dir (str): Directory to save the trained model
        export_onnx (bool): Whether to export to ONNX format
        evaluate_consistency (bool): Whether to evaluate XGBoost vs ONNX consistency
    
    Returns:
        tuple: (trained_model, label_encoder, test_accuracy, onnx_consistency)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # -----------------------------
    # Load & preprocess
    # -----------------------------
    preprocessor = NIDSPreprocessor("scaler_params.json")
    X, labels = preprocessor.preprocess_data(
        Path(data_dir) / dataset,
        fit_scaler=False
    )

    # Remove benign
    mask = labels != "BENIGN"
    X_attack = X[mask]
    y_attack = labels[mask]

    print(f"Attack samples: {len(X_attack)}")
    print("Attack classes:", np.unique(y_attack))

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
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
    )

    print("Training XGBoost classifier...")
    model.fit(X_train, y_train)

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
            output_path=onnx_path
        )
        
        # Evaluate consistency if requested
        if evaluate_consistency:
            print("Evaluating XGBoost vs ONNX consistency...")
            max_diff, mean_diff, _, _ = evaluate_xgb_onnx_consistency(
                model_path=model_path,
                onnx_path=onnx_path,
                data_dir=data_dir,
                dataset=dataset,
                num_samples=100
            )
            
            onnx_consistency = {
                "max_difference": max_diff,
                "mean_difference": mean_diff
            }
            
            print(f"Max probability difference: {max_diff:.6e}")
            print(f"Mean probability difference: {mean_diff:.6e}")
            
            # Quality assessment
            if max_diff < 1e-6:
                print("Excellent ONNX conversion (diff < 1e-6)")
            elif max_diff < 1e-4:
                print("Good ONNX conversion (diff < 1e-4)")
            else:
                print("Poor ONNX conversion (diff >= 1e-4)")

    return model, encoder, model.score(X_test, y_test), onnx_consistency