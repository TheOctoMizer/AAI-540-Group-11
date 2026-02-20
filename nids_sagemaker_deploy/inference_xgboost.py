"""
Custom inference script for XGBoost classifier on SageMaker.

This script handles:
1. Model loading (ONNX format)
2. Input preprocessing
3. Classification
4. Output formatting with attack labels
"""

import json
import os
import numpy as np
import onnxruntime as ort
from typing import Dict, List


def model_fn(model_dir):
    """
    Load the ONNX model for inference.
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Dictionary with ONNX session and label map
    """
    # Load ONNX model
    model_path = os.path.join(model_dir, "xgb_classifier.onnx")
    session = ort.InferenceSession(model_path)
    
    # Load label map
    label_map_path = os.path.join(model_dir, "xgb_label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    
    # Convert string keys to int
    label_map = {int(k): v for k, v in label_map.items()}
    
    print(f"Model loaded successfully")
    print(f"Input name: {session.get_inputs()[0].name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Number of classes: {len(label_map)}")
    print(f"Classes: {list(label_map.values())}")
    
    return {
        "session": session,
        "label_map": label_map
    }


def input_fn(request_body, content_type="application/json"):
    """
    Deserialize and prepare the input data.
    
    Args:
        request_body: The request payload
        content_type: The content type of the request
        
    Returns:
        Preprocessed input array
    """
    if content_type == "application/json":
        data = json.loads(request_body)
        
        # Support both single sample and batch
        if isinstance(data, list):
            # Batch of samples
            features = np.array(data, dtype=np.float32)
        elif isinstance(data, dict) and "encoded" in data:
            # Encoded features from autoencoder
            features = np.array(data["encoded"], dtype=np.float32)
            if features.ndim == 1:
                features = features.reshape(1, -1)
        else:
            raise ValueError("Invalid input format. Expected list or dict with 'encoded' key.")
        
        return features
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make predictions with the model.
    
    Args:
        input_data: Preprocessed input array
        model: Dictionary with session and label_map
        
    Returns:
        Dictionary with predictions
    """
    session = model["session"]
    label_map = model["label_map"]
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    # Parse outputs
    # outputs[0] is typically class IDs (int64)
    # outputs[1] is typically probabilities (float32)
    
    if len(outputs) >= 2:
        class_ids = outputs[0].flatten()
        probabilities = outputs[1]
    else:
        # If only one output, assume it's probabilities
        probabilities = outputs[0]
        class_ids = np.argmax(probabilities, axis=1)
    
    # Convert class IDs to labels
    labels = [label_map.get(int(cid), f"Class_{cid}") for cid in class_ids]
    
    # Get confidence scores
    if probabilities.ndim == 2:
        # Multi-class probabilities
        confidences = np.max(probabilities, axis=1)
    else:
        # Binary or single output
        confidences = probabilities.flatten()
    
    return {
        "class_ids": class_ids.tolist(),
        "labels": labels,
        "confidences": confidences.tolist(),
        "probabilities": probabilities.tolist() if probabilities.ndim == 2 else None
    }


def output_fn(prediction, accept="application/json"):
    """
    Serialize the prediction output.
    
    Args:
        prediction: Model prediction dictionary
        accept: Requested response content type
        
    Returns:
        Serialized response
    """
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
