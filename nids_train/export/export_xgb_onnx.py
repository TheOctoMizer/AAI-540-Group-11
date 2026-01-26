from pathlib import Path
import json

import onnxmltools
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType


def export_xgb_to_onnx(model_path, label_map_path, output_path, input_dim=77):
    """
    Export XGBoost model to ONNX format.
    
    Args:
        model_path (Path): Path to XGBoost model file
        label_map_path (Path): Path to label mapping file
        output_path (Path): Path to save ONNX model
        input_dim (int): Input dimension (default: 77)
    
    Returns:
        Path: Path to the saved ONNX model
    """
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(label_map_path) as f:
        label_map = json.load(f)

    initial_type = [("input", FloatTensorType([None, input_dim]))]

    onnx_model = onnxmltools.convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=15
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"XGBoost ONNX model saved to {output_path}")
    print(f"Classes: {label_map}")
    
    return output_path


def main():
    """Standalone script entry point."""
    MODEL_DIR = Path("./model")
    
    export_xgb_to_onnx(
        model_path=MODEL_DIR / "xgb_classifier.json",
        label_map_path=MODEL_DIR / "xgb_label_map.json",
        output_path=MODEL_DIR / "xgb_classifier.onnx"
    )


if __name__ == "__main__":
    main()