"""
Custom inference script for Autoencoder on SageMaker.

This script handles:
1. Model loading
2. Input preprocessing
3. Reconstruction
4. MSE error calculation
5. Output formatting
"""

import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class Autoencoder(nn.Module):
    """Autoencoder architecture matching training."""
    
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        """Return encoded representation."""
        return self.encoder(x)


def model_fn(model_dir):
    """
    Load the model for inference.
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Loaded PyTorch model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model metadata
    with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    input_dim = metadata["input_dim"]
    latent_dim = metadata.get("latent_dim", 8)
    
    # Initialize model
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    
    # Load weights
    model_path = os.path.join(model_dir, "autoencoder.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Input dim: {input_dim}, Latent dim: {latent_dim}")
    
    return model


def input_fn(request_body, content_type="application/json"):
    """
    Deserialize and prepare the input data.
    
    Args:
        request_body: The request payload
        content_type: The content type of the request
        
    Returns:
        Preprocessed input tensor
    """
    if content_type == "application/json":
        data = json.loads(request_body)
        
        # Support both single sample and batch
        if isinstance(data, list):
            # Batch of samples
            features = np.array(data, dtype=np.float32)
        elif isinstance(data, dict) and "features" in data:
            # Single sample with key
            features = np.array(data["features"], dtype=np.float32)
            if features.ndim == 1:
                features = features.reshape(1, -1)
        else:
            raise ValueError("Invalid input format. Expected list or dict with 'features' key.")
        
        return torch.FloatTensor(features)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make predictions with the model.
    
    Args:
        input_data: Preprocessed input tensor
        model: Loaded model
        
    Returns:
        Dictionary with reconstruction and error
    """
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    with torch.no_grad():
        # Get reconstruction
        reconstruction = model(input_data)
        
        # Get encoded representation
        encoded = model.encode(input_data)
        
        # Calculate MSE error per sample
        mse_error = torch.mean((input_data - reconstruction) ** 2, dim=1)
    
    return {
        "reconstruction": reconstruction.cpu().numpy(),
        "encoded": encoded.cpu().numpy(),
        "mse_error": mse_error.cpu().numpy()
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
        # Convert numpy arrays to lists for JSON serialization
        response = {
            "reconstruction": prediction["reconstruction"].tolist(),
            "encoded": prediction["encoded"].tolist(),
            "mse_error": prediction["mse_error"].tolist()
        }
        return json.dumps(response)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
