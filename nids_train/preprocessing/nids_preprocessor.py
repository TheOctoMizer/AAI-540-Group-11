"""
Shared preprocessing utilities for NIDS training and inference.
Ensures consistent data processing across all scripts.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class NIDSPreprocessor:
    """Handles data preprocessing with consistent scaler management."""
    
    def __init__(self, scaler_path="scaler_params.json"):
        self.scaler_path = scaler_path
        self.scaler = None
        self.feature_columns = None
        
    def fit_scaler(self, data_path):
        """Fit scaler on training data and save parameters."""
        print(f"Fitting scaler on training data: {data_path}...")
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        # Filter benign traffic (same as training)
        df = df[df["Label"] == "BENIGN"]
        
        # Drop identifier and redundant/zero-variance columns
        drop_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp", "Label",
            "Avg Bwd Segment Size", "Avg Fwd Segment Size", "Subflow Bwd Packets", 
            "Fwd Header Length.1", "SYN Flag Count", "Subflow Bwd Bytes", 
            "ECE Flag Count", "Subflow Fwd Packets", "Subflow Fwd Bytes",
            "Bwd Avg Bulk Rate", "Bwd Avg Packets/Bulk", "Bwd Avg Bytes/Bulk", 
            "Fwd Avg Bulk Rate", "Fwd Avg Packets/Bulk", "Fwd Avg Bytes/Bulk"
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
            
        print(f"Scaler fitted and saved to {self.scaler_path}")
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
            
            print(f"Scaler loaded from {self.scaler_path}")
            return self.scaler
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}. Run training first.")

    def aggregate_benign_data(self, data_dir, fit_scaler=True):
        """
        Iterate through all CSV files in data_dir, extract benign samples,
        and return a combined scaled NumPy array.
        """
        data_dir = Path(data_dir)
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        all_benign_dfs = []
        
        # Drop identifier and redundant/zero-variance columns
        drop_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp", "Label",
            "Avg Bwd Segment Size", "Avg Fwd Segment Size", "Subflow Bwd Packets", 
            "Fwd Header Length.1", "SYN Flag Count", "Subflow Bwd Bytes", 
            "ECE Flag Count", "Subflow Fwd Packets", "Subflow Fwd Bytes",
            "Bwd Avg Bulk Rate", "Bwd Avg Packets/Bulk", "Bwd Avg Bytes/Bulk", 
            "Fwd Avg Bulk Rate", "Fwd Avg Packets/Bulk", "Fwd Avg Bytes/Bulk"
        ]

        print(f"Aggregating benign data from {len(csv_files)} files...")
        
        for f in csv_files:
            print(f"  Processing {f.name}...")
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            
            # Keep only benign
            df = df[df["Label"] == "BENIGN"]
            if len(df) == 0:
                continue
                
            # Clean data (remove nan/inf) BEFORE dropping Label
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in df.columns if c != "Label"])
            
            if len(df) == 0:
                continue

            # Drop identifiers and redundant cols
            df = df.drop(columns=[c for c in drop_cols if c in df.columns and c != "Label"], errors="ignore")
            
            # Identify feature columns (everything except Label)
            features_in_this_file = [c for c in df.columns if c != "Label"]
            
            if self.feature_columns is None:
                self.feature_columns = features_in_this_file
            
            # Ensure consistent column ordering and keep features only
            df_features = df[self.feature_columns]
            all_benign_dfs.append(df_features)
            
        if not all_benign_dfs:

            raise ValueError("No benign samples found in any files.")
            
        combined_df = pd.concat(all_benign_dfs, ignore_index=True)
        print(f"Total benign samples aggregated: {len(combined_df):,}")
        
        # Handle scaler
        if fit_scaler:
            print("Fitting scaler on aggregated benign data...")
            self.scaler = MinMaxScaler()
            self.scaler.fit(combined_df.values)
            
            # Save parameters
            scaler_params = {
                "scale": self.scaler.scale_.tolist(),
                "min": self.scaler.min_.tolist(),
                "n_features": self.scaler.n_features_in_,
                "feature_columns": self.feature_columns
            }
            with open(self.scaler_path, "w") as f:
                json.dump(scaler_params, f, indent=2)
            print(f"Scaler saved to {self.scaler_path}")
        elif self.scaler is None:
            self.load_scaler()
            
        X = self.scaler.transform(combined_df.values).astype(np.float32)
        return X

    def aggregate_attack_data(self, data_dir):
        """
        Iterate through all CSV files in data_dir, extract ALL attack samples (Label != BENIGN),
        and return combined features and labels.
        """
        data_dir = Path(data_dir)
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        all_features = []
        all_labels = []
        
        # Drop identifier and redundant/zero-variance columns
        drop_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp", "Label",
            "Avg Bwd Segment Size", "Avg Fwd Segment Size", "Subflow Bwd Packets", 
            "Fwd Header Length.1", "SYN Flag Count", "Subflow Bwd Bytes", 
            "ECE Flag Count", "Subflow Fwd Packets", "Subflow Fwd Bytes",
            "Bwd Avg Bulk Rate", "Bwd Avg Packets/Bulk", "Bwd Avg Bytes/Bulk", 
            "Fwd Avg Bulk Rate", "Fwd Avg Packets/Bulk", "Fwd Avg Bytes/Bulk"
        ]

        print(f"Aggregating attack data from {len(csv_files)} files...")
        
        for f in csv_files:
            print(f"  Processing {f.name}...")
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            
            # Keep only attacks
            df = df[df["Label"] != "BENIGN"]
            if len(df) == 0:
                continue
                
            # Clean data (remove nan/inf) BEFORE dropping Label
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in df.columns if c != "Label"])
            
            if len(df) == 0:
                continue

            # Extract labels from the cleaned subset
            labels = df["Label"].values
            
            # Drop identifiers and redundant cols
            df = df.drop(columns=[c for c in drop_cols if c in df.columns and c != "Label"], errors="ignore")
            
            # Ensure consistent column ordering
            if self.feature_columns is not None:
                df = df[self.feature_columns]
            else:
                 self.feature_columns = [c for c in df.columns if c != "Label"]
                 df = df[self.feature_columns]
                
            all_features.append(df.values)
            all_labels.append(labels)
            
        if not all_features:

            raise ValueError("No attack samples found in any files.")
            
        X = np.vstack(all_features).astype(np.float32)
        y = np.concatenate(all_labels)
        
        print(f"Total attack samples aggregated: {len(X):,}")
        
        # Transform data using loaded/fitted scaler
        if self.scaler is None:
            self.load_scaler()
            
        X_scaled = self.scaler.transform(X).astype(np.float32)
        return X_scaled, y

    def preprocess_data(self, data_path, fit_scaler=False):
        """Preprocess data with consistent feature handling."""
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        
        # Store original labels for evaluation
        labels = df.get("Label", pd.Series([None] * len(df)))
        
        # Drop identifier and redundant/zero-variance columns
        drop_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp", "Label",
            "Avg Bwd Segment Size", "Avg Fwd Segment Size", "Subflow Bwd Packets", 
            "Fwd Header Length.1", "SYN Flag Count", "Subflow Bwd Bytes", 
            "ECE Flag Count", "Subflow Fwd Packets", "Subflow Fwd Bytes",
            "Bwd Avg Bulk Rate", "Bwd Avg Packets/Bulk", "Bwd Avg Bytes/Bulk", 
            "Fwd Avg Bulk Rate", "Fwd Avg Packets/Bulk", "Fwd Avg Bytes/Bulk"
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        
        # Store original length before cleaning
        original_length = len(df)
        
        # Clean data (remove infinite values and NaN)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Filter labels to match cleaned data
        if len(df) < original_length:
            # Get the indices of rows that were kept
            kept_indices = df.index
            labels = labels.iloc[kept_indices]
        
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
