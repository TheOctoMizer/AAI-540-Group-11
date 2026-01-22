"""
Shared preprocessing utilities for NIDS training and inference.
Ensures consistent data processing across all scripts.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class NIDSPreprocessor:
    """Handles data preprocessing with consistent scaler management."""
    
    def __init__(self, scaler_path="scaler_params.json"):
        self.scaler_path = scaler_path
        self.scaler = None
        self.feature_columns = None
        
    def fit_scaler(self, data_path):
        """Fit scaler on training data and save parameters."""
        print(f"⏳ Fitting scaler on training data: {data_path}...")
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        # Filter benign traffic (same as training)
        df = df[df["Label"] == "BENIGN"]
        
        # Drop identifier columns
        drop_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp", "Label"
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Store feature order for consistency
        self.feature_columns = df.columns.tolist()
        
        # Fit scaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(df.values)
        
        # Save parameters
        scaler_params = {
            "scale": self.scaler.scale_.tolist(),
            "min": self.scaler.min_.tolist(),
            "n_features": self.scaler.n_features_in_,
            "feature_columns": self.feature_columns
        }
        
        with open(self.scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
            
        print(f"✅ Scaler fitted and saved to {self.scaler_path}")
        return self.scaler
    
    def load_scaler(self):
        """Load saved scaler parameters."""
        try:
            with open(self.scaler_path, "r") as f:
                params = json.load(f)
            
            self.scaler = MinMaxScaler()
            self.scaler.scale_ = np.array(params["scale"])
            self.scaler.min_ = np.array(params["min"])
            self.scaler.n_features_in_ = params["n_features"]
            self.feature_columns = params.get("feature_columns")
            
            print(f"✅ Scaler loaded from {self.scaler_path}")
            return self.scaler
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}. Run export_scaler.py first.")
    
    def preprocess_data(self, data_path, fit_scaler=False):
        """Preprocess data with consistent feature handling."""
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        # Store original labels for evaluation
        labels = df.get("Label", pd.Series([None] * len(df)))
        
        # Drop identifier columns
        drop_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp", "Label"
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Ensure consistent feature order
        if self.feature_columns is not None:
            # Reorder columns to match training
            df = df[self.feature_columns]
        
        # Handle scaler
        if fit_scaler:
            self.fit_scaler(data_path)
        elif self.scaler is None:
            self.load_scaler()
        
        # Transform data
        X = self.scaler.transform(df.values).astype(np.float32)
        
        return X, labels.values


def load_and_preprocess_data(path, preprocessor=None, fit_scaler=False):
    """Legacy function for backward compatibility."""
    if preprocessor is None:
        preprocessor = NIDSPreprocessor()
    
    X, labels = preprocessor.preprocess_data(path, fit_scaler=fit_scaler)
    return X, labels
