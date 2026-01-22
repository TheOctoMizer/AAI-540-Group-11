"""
NIDS Training Script - Consolidated training and asset export.

This script handles:
1. Model training with autoencoder architecture
2. Automatic threshold calculation
3. Asset export for production (PyTorch, ONNX, scaler, thresholds)
4. Test sample generation for validation
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import NIDSPreprocessor


class Autoencoder(nn.Module):
    """Autoencoder for network intrusion detection."""
    
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder: Compress features down to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # Latent Space
            nn.ReLU(),
        )

        # Decoder: Reconstruct original input from latent space
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),  # Features are 0-1 scaled
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class NIDSTrainer:
    """Consolidated NIDS training and asset export."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.preprocessor = NIDSPreprocessor(config.scaler_path)
        self.model = None
        self.input_dim = None
        
        print(f"Using device: {self.device}")
    
    def load_and_prepare_data(self):
        """Load and preprocess training data."""
        print(f"Loading training data: {self.config.dataset_path}")
        
        X, labels = self.preprocessor.preprocess_data(self.config.dataset_path, fit_scaler=True)
        
        # Filter only benign samples for training
        benign_mask = labels == "BENIGN"
        X_benign = X[benign_mask]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Benign samples: {X_benign.shape[0]} ({X_benign.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"Features: {X.shape[1]}")
        
        # Split data
        X_train, X_val = train_test_split(
            X_benign, 
            test_size=self.config.val_split, 
            random_state=self.config.random_state
        )
        
        self.input_dim = X.shape[1]
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        
        return torch.FloatTensor(X_train), torch.FloatTensor(X_val)
    
    def create_model(self):
        """Initialize and return the autoencoder model."""
        self.model = Autoencoder(self.input_dim).to(self.device)
        return self.model
    
    def train_model(self, X_train, X_val):
        """Train the autoencoder model."""
        print(f"\nTraining: {self.config.epochs} epochs, batch_size={self.config.batch_size}, lr={self.config.learning_rate}")
        print(f"Model parameters: {sum(p.numel() for p in Autoencoder(self.input_dim).parameters())}")
        print(f"Device: {self.device}")
        
        # Data loaders
        train_loader = DataLoader(
            TensorDataset(X_train), 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val), 
            batch_size=self.config.batch_size
        )
        
        # Model and optimizer
        model = self.create_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Training history
        train_history = []
        val_errors = []
        best_val_loss = float('inf')
        
        print(f"{'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>9} | {'Time':>5} | {'LR':>8} | {'Improvement'}")
        print(f"{'-'*5}-+-{'-'*11}-+-{'-'*9}-+-{'-'*5}-+-{'-'*8}-+-{'-'*12}")
        
        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0
            start_time = time.time()

            # Training phase
            for batch in train_loader:
                inputs = batch[0].to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0
            epoch_val_errors = []
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()
                    
                    # Calculate per-sample reconstruction errors
                    mse_per_sample = nn.functional.mse_loss(outputs, inputs, reduction='none')
                    mse_per_sample = mse_per_sample.mean(dim=1)
                    epoch_val_errors.extend(mse_per_sample.cpu().numpy())

            # Record metrics
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Check for improvement
            improvement = val_loss_avg < best_val_loss
            if improvement:
                best_val_loss = val_loss_avg
                
            train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'time': epoch_time,
                'learning_rate': current_lr
            })
            
            val_errors.extend(epoch_val_errors)

            print(f"{epoch + 1:5d} | {train_loss_avg:11.6f} | {val_loss_avg:9.6f} | {epoch_time:5.1f}s | {current_lr:8.2e} | {'✓' if improvement else '-'}")
        print(f"\nBest validation loss: {best_val_loss:.6f}")
        print(f"Total training time: {sum(h['time'] for h in train_history):.1f}s")
        
        return model, train_history, val_errors
    
    def calculate_thresholds(self, val_errors):
        """Calculate detection thresholds from validation errors."""
        print(f"\nCalculating detection thresholds from {len(val_errors)} validation samples...")
        
        val_errors = np.array(val_errors)
        thresholds = {
            "mean": float(val_errors.mean()),
            "std": float(val_errors.std()),
            "p90": float(np.percentile(val_errors, 90)),
            "p95": float(np.percentile(val_errors, 95)),
            "p99": float(np.percentile(val_errors, 99)),
            "p999": float(np.percentile(val_errors, 99.9)),
            "max": float(val_errors.max())
        }
        
        # Save thresholds
        with open(self.config.threshold_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        
        print(f"Thresholds saved to {self.config.threshold_path}")
        print(f"Error statistics: mean={thresholds['mean']:.6f}, std={thresholds['std']:.6f}")
        print(f"Percentiles: 90th={thresholds['p90']:.6f}, 95th={thresholds['p95']:.6f}, 99th={thresholds['p99']:.6f}")
        
        return thresholds
    
    def export_pytorch_model(self, model):
        """Export PyTorch model."""
        model_size = self.config.pytorch_path.stat().st_size if hasattr(self.config.pytorch_path, 'stat') else 0
        torch.save(model.state_dict(), self.config.pytorch_path)
        print(f"PyTorch model saved: {self.config.pytorch_path}")
    
    def export_onnx_model(self, model):
        """Export ONNX model for production."""
        print(f"Exporting ONNX model...")
        
        dummy_input = torch.randn(1, self.input_dim).to(self.device)
        torch.onnx.export(
            model,
            dummy_input,
            self.config.onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        
        # Get file size
        onnx_size = self.config.onnx_path.stat().st_size if hasattr(self.config.onnx_path, 'stat') else 0
        print(f"ONNX model saved: {self.config.onnx_path} ({onnx_size/1024:.1f} KB)")
    
    def export_test_samples(self, num_samples=100):
        """Export test samples for validation."""
        print(f"Exporting {num_samples} test samples...")
        
        # Load some benign samples
        X, labels = self.preprocessor.preprocess_data(self.config.dataset_path)
        benign_mask = labels == "BENIGN"
        X_benign = X[benign_mask][:num_samples]
        
        test_samples = {
            "samples": X_benign.tolist(),
            "num_samples": len(X_benign),
            "features": self.input_dim,
            "scaler_params": self.config.scaler_path
        }
        
        with open(self.config.test_samples_path, "w") as f:
            json.dump(test_samples, f, indent=2)
        
        print(f"Test samples saved: {self.config.test_samples_path}")
    
    def export_training_metadata(self, train_history, thresholds):
        """Export training metadata and configuration."""
        metadata = {
            "config": {
                "dataset_path": self.config.dataset_path,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "val_split": self.config.val_split,
                "input_features": self.input_dim,
                "device": str(self.device)
            },
            "training_history": train_history,
            "final_thresholds": thresholds,
            "model_architecture": {
                "encoder_layers": [self.input_dim, 64, 32, 8],
                "decoder_layers": [8, 32, 64, self.input_dim],
                "latent_dim": 8,
                "total_parameters": sum(p.numel() for p in Autoencoder(self.input_dim).parameters())
            },
            "performance": {
                "best_val_loss": min(h['val_loss'] for h in train_history),
                "final_train_loss": train_history[-1]['train_loss'],
                "final_val_loss": train_history[-1]['val_loss'],
                "total_training_time": sum(h['time'] for h in train_history)
            },
            "assets_generated": {
                "pytorch_model": self.config.pytorch_path,
                "onnx_model": self.config.onnx_path,
                "scaler_params": self.config.scaler_path,
                "thresholds": self.config.threshold_path,
                "test_samples": self.config.test_samples_path
            }
        }
        
        with open(self.config.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training metadata saved: {self.config.metadata_path}")
    
    def train(self):
        """Main training pipeline."""
        print("NIDS Training Pipeline")
        print("=" * 50)
        
        # 1. Load and prepare data
        X_train, X_val = self.load_and_prepare_data()
        
        # 2. Train model
        model, train_history, val_errors = self.train_model(X_train, X_val)
        
        # 3. Calculate thresholds
        thresholds = self.calculate_thresholds(val_errors)
        
        # 4. Export assets
        print(f"\nExporting production assets...")
        self.export_pytorch_model(model)
        self.export_onnx_model(model)
        self.export_test_samples()
        
        # 5. Export metadata
        self.export_training_metadata(train_history, thresholds)
        
        print(f"\nTraining completed successfully")
        print(f"Assets ready for production deployment")


class Config:
    """Training configuration."""
    
    def __init__(self, args):
        self.dataset_path = args.dataset
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.val_split = args.val_split
        self.random_state = 42
        
        # Output paths
        self.pytorch_path = args.pytorch_output
        self.onnx_path = args.onnx_output
        self.scaler_path = args.scaler_output
        self.threshold_path = args.threshold_output
        self.test_samples_path = args.test_samples_output
        self.metadata_path = args.metadata_output


def main():
    parser = argparse.ArgumentParser(description="NIDS Training Pipeline")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, 
                       default="./nids_dataset/Monday-WorkingHours.pcap_ISCX.csv",
                       help="Path to training dataset")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    
    # Output arguments
    parser.add_argument("--pytorch-output", type=str, default="nids_autoencoder.pt",
                       help="PyTorch model output path")
    parser.add_argument("--onnx-output", type=str, default="nids_autoencoder.onnx",
                       help="ONNX model output path")
    parser.add_argument("--scaler-output", type=str, default="scaler_params.json",
                       help="Scaler parameters output path")
    parser.add_argument("--threshold-output", type=str, default="detection_threshold.json",
                       help="Detection thresholds output path")
    parser.add_argument("--test-samples-output", type=str, default="test_samples.json",
                       help="Test samples output path")
    parser.add_argument("--metadata-output", type=str, default="training_metadata.json",
                       help="Training metadata output path")
    
    args = parser.parse_args()
    config = Config(args)
    
    # Validate dataset exists
    if not Path(config.dataset_path).exists():
        print(f"❌ Dataset not found: {config.dataset_path}")
        print("Please ensure the CIC-IDS2017 dataset is downloaded and in the correct location")
        return
    
    # Run training
    trainer = NIDSTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
