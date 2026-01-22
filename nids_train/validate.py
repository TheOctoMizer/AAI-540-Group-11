"""
NIDS Validation Script - Model validation and threshold optimization.

This script handles:
1. Cross-validation for model robustness
2. Threshold optimization and analysis
3. Model comparison and selection
4. Validation on multiple datasets
5. Statistical significance testing
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort

from preprocessing import NIDSPreprocessor
from train import Autoencoder


class NIDSValidator:
    """Comprehensive NIDS model validation and threshold optimization."""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = NIDSPreprocessor(config.scaler_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def load_model(self, input_dim: int, model_path: str = None):
        """Load model for validation."""
        model_path = model_path or self.config.model_path
        
        if self.config.model_type.lower() == "pytorch":
            model = Autoencoder(input_dim)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model
        elif self.config.model_type.lower() == "onnx":
            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            return session
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def predict_reconstruction_errors(self, model, X: np.ndarray):
        """Get reconstruction errors from model."""
        if self.config.model_type.lower() == "pytorch":
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = model(X_tensor)
                mse = nn.functional.mse_loss(outputs, X_tensor, reduction='none')
                return mse.mean(dim=1).cpu().numpy()
        else:  # ONNX
            batch_size = self.config.batch_size
            errors = []
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                outputs = model.run([output_name], {input_name: batch})[0]
                mse = np.mean((outputs - batch) ** 2, axis=1)
                errors.extend(mse)
            
            return np.array(errors)
    
    def cross_validation(self, data_path: str, n_folds: int = 5) -> Dict[str, Any]:
        """Perform k-fold cross-validation."""
        print(f"\n🔄 Performing {n_folds}-fold cross-validation...")
        
        # Load and preprocess data
        X, labels = self.preprocessor.preprocess_data(data_path, fit_scaler=True)
        
        # Filter benign samples for training validation
        benign_mask = labels == "BENIGN"
        X_benign = X[benign_mask]
        
        print(f"Benign samples for CV: {len(X_benign)}")
        
        # Setup cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_results = {
            "fold_results": [],
            "mean_performance": {},
            "std_performance": {}
        }
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_benign)):
            print(f"\n📂 Fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train_fold = X_benign[train_idx]
            X_val_fold = X_benign[val_idx]
            
            # Train model on fold
            model = self._train_fold_model(X_train_fold, X_val_fold)
            
            # Evaluate on validation fold
            val_errors = self.predict_reconstruction_errors(model, X_val_fold)
            
            # Calculate metrics
            fold_metrics.append({
                "fold": fold + 1,
                "train_samples": len(X_train_fold),
                "val_samples": len(X_val_fold),
                "mean_error": float(np.mean(val_errors)),
                "std_error": float(np.std(val_errors)),
                "p95_error": float(np.percentile(val_errors, 95)),
                "p99_error": float(np.percentile(val_errors, 99))
            })
            
            print(f"   Mean Error: {fold_metrics[-1]['mean_error']:.6f}")
            print(f"   95th Percentile: {fold_metrics[-1]['p95_error']:.6f}")
        
        # Calculate aggregate statistics
        cv_results["fold_results"] = fold_metrics
        
        for metric in ["mean_error", "std_error", "p95_error", "p99_error"]:
            values = [fold[metric] for fold in fold_metrics]
            cv_results["mean_performance"][metric] = float(np.mean(values))
            cv_results["std_performance"][metric] = float(np.std(values))
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"   Mean Error: {cv_results['mean_performance']['mean_error']:.6f} ± {cv_results['std_performance']['mean_error']:.6f}")
        print(f"   95th Percentile: {cv_results['mean_performance']['p95_error']:.6f} ± {cv_results['std_performance']['p95_error']:.6f}")
        print(f"   99th Percentile: {cv_results['mean_performance']['p99_error']:.6f} ± {cv_results['std_performance']['p99_error']:.6f}")
        
        return cv_results
    
    def _train_fold_model(self, X_train, X_val):
        """Train model for a single fold."""
        # Create fresh model
        model = Autoencoder(X_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        
        # Train for fewer epochs in CV
        epochs = 5
        batch_size = 512
        
        for epoch in range(epochs):
            model.train()
            
            # Simple training loop
            for i in range(0, len(X_train_tensor), batch_size):
                batch = X_train_tensor[i:i+batch_size].to(self.device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        return model
    
    def threshold_optimization(self, data_path: str) -> Dict[str, Any]:
        """Optimize detection thresholds using various methods."""
        print(f"\n🎯 Optimizing detection thresholds...")
        
        # Load data
        X, labels = self.preprocessor.preprocess_data(data_path)
        binary_labels = np.array([1 if label != "BENIGN" else 0 for label in labels])
        
        # Load model
        model = self.load_model(X.shape[1])
        errors = self.predict_reconstruction_errors(model, X)
        
        optimization_results = {
            "methods": {},
            "recommended_threshold": None,
            "method_performance": {}
        }
        
        # Method 1: Percentile-based thresholds
        percentiles = [90, 95, 99, 99.5, 99.9]
        percentile_results = {}
        
        for percentile in percentiles:
            threshold = np.percentile(errors, percentile)
            predictions = errors > threshold
            
            tn, fp, fn, tp = confusion_matrix(binary_labels, predictions).ravel()
            
            metrics = {
                "threshold": float(threshold),
                "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
                "precision": float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                "recall": float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                "f1_score": float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
                "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0)
            }
            
            percentile_results[f"p{percentile}"] = metrics
        
        optimization_results["methods"]["percentile"] = percentile_results
        
        # Method 2: Youden's J statistic (maximizing sensitivity + specificity - 1)
        if len(np.unique(binary_labels)) > 1:
            fpr, tpr, thresholds_roc = roc_curve(binary_labels, errors)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold_roc = thresholds_roc[optimal_idx]
            
            predictions = errors > optimal_threshold_roc
            tn, fp, fn, tp = confusion_matrix(binary_labels, predictions).ravel()
            
            youden_metrics = {
                "threshold": float(optimal_threshold_roc),
                "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
                "precision": float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                "recall": float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                "f1_score": float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
                "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
                "youden_j": float(youden_j[optimal_idx])
            }
            
            optimization_results["methods"]["youden_j"] = youden_metrics
        
        # Method 3: Maximum F1-score
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(binary_labels, errors)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold_f1 = thresholds_pr[optimal_idx]
        
        predictions = errors > optimal_threshold_f1
        tn, fp, fn, tp = confusion_matrix(binary_labels, predictions).ravel()
        
        f1_metrics = {
            "threshold": float(optimal_threshold_f1),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "precision": float(tp / (tp + fp) if (tp + fp) > 0 else 0),
            "recall": float(tp / (tp + fn) if (tp + fn) > 0 else 0),
            "f1_score": float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
            "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0)
        }
        
        optimization_results["methods"]["max_f1"] = f1_metrics
        
        # Recommend best threshold based on F1-score
        best_method = "max_f1"
        best_threshold = optimization_results["methods"][best_method]["threshold"]
        optimization_results["recommended_threshold"] = {
            "method": best_method,
            "threshold": best_threshold,
            "f1_score": optimization_results["methods"][best_method]["f1_score"]
        }
        
        # Print results
        print(f"\n📊 Threshold Optimization Results:")
        print(f"   Method       | Threshold | F1-Score | Recall | FPR")
        print(f"   -------------|-----------|----------|--------|-----")
        
        for method_name, method_results in optimization_results["methods"].items():
            print(f"   {method_name:12s} | {method_results['threshold']:9.6f} | "
                  f"{method_results['f1_score']:8.4f} | {method_results['recall']:6.4f} | "
                  f"{method_results['false_positive_rate']:5.4f}")
        
        print(f"\n🎯 Recommended: {best_method} with threshold {best_threshold:.6f}")
        
        return optimization_results
    
    def multi_dataset_validation(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Validate model across multiple datasets."""
        print(f"\n🌐 Validating across {len(dataset_paths)} datasets...")
        
        multi_results = {
            "dataset_results": {},
            "aggregate_performance": {}
        }
        
        all_metrics = []
        
        for dataset_path in dataset_paths:
            dataset_name = Path(dataset_path).name
            print(f"\n📂 Validating on: {dataset_name}")
            
            try:
                # Load and preprocess
                X, labels = self.preprocessor.preprocess_data(dataset_path)
                binary_labels = np.array([1 if label != "BENIGN" else 0 for label in labels])
                
                # Load model
                model = self.load_model(X.shape[1])
                errors = self.predict_reconstruction_errors(model, X)
                
                # Use 99th percentile as threshold
                threshold = np.percentile(errors, 99)
                predictions = errors > threshold
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(binary_labels, predictions).ravel()
                
                metrics = {
                    "total_samples": len(X),
                    "benign_samples": np.sum(binary_labels == 0),
                    "attack_samples": np.sum(binary_labels == 1),
                    "threshold": float(threshold),
                    "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
                    "precision": float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                    "recall": float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                    "f1_score": float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
                    "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
                    "mean_error": float(np.mean(errors))
                }
                
                multi_results["dataset_results"][dataset_name] = metrics
                all_metrics.append(metrics)
                
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   F1-Score: {metrics['f1_score']:.4f}")
                print(f"   Detection Rate: {metrics['recall']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error processing {dataset_name}: {str(e)}")
                continue
        
        # Calculate aggregate statistics
        if all_metrics:
            for metric in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
                values = [m[metric] for m in all_metrics]
                multi_results["aggregate_performance"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            
            print(f"\n📊 Aggregate Performance Across Datasets:")
            agg = multi_results["aggregate_performance"]
            print(f"   Accuracy:  {agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}")
            print(f"   Precision: {agg['precision']['mean']:.4f} ± {agg['precision']['std']:.4f}")
            print(f"   Recall:    {agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}")
            print(f"   F1-Score:  {agg['f1_score']['mean']:.4f} ± {agg['f1_score']['std']:.4f}")
        
        return multi_results
    
    def save_validation_results(self, results: Dict[str, Any], output_path: str):
        """Save validation results to file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Validation results saved to {output_path}")
    
    def validate(self, dataset_path: str = None, dataset_paths: List[str] = None,
                do_cross_validation: bool = False, do_threshold_optimization: bool = False,
                do_multi_dataset: bool = False):
        """Main validation pipeline."""
        print("🚀 Starting NIDS Validation Pipeline")
        print("=" * 50)
        
        # Load preprocessor
        self.preprocessor.load_scaler()
        
        validation_results = {
            "validation_config": {
                "model_path": self.config.model_path,
                "model_type": self.config.model_type,
                "scaler_path": self.config.scaler_path,
                "batch_size": self.config.batch_size
            }
        }
        
        # Cross-validation
        if do_cross_validation and dataset_path:
            cv_results = self.cross_validation(dataset_path, n_folds=5)
            validation_results["cross_validation"] = cv_results
        
        # Threshold optimization
        if do_threshold_optimization and dataset_path:
            threshold_results = self.threshold_optimization(dataset_path)
            validation_results["threshold_optimization"] = threshold_results
        
        # Multi-dataset validation
        if do_multi_dataset and dataset_paths:
            multi_results = self.multi_dataset_validation(dataset_paths)
            validation_results["multi_dataset_validation"] = multi_results
        
        # Save results
        self.save_validation_results(validation_results, self.config.output_path)
        
        return validation_results


class Config:
    """Validation configuration."""
    
    def __init__(self, args):
        self.model_path = args.model
        self.model_type = args.type
        self.scaler_path = args.scaler
        self.batch_size = args.batch_size
        self.output_path = args.output


def main():
    parser = argparse.ArgumentParser(description="NIDS Validation Pipeline")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="nids_autoencoder.onnx",
                       help="Path to model file")
    parser.add_argument("--type", type=str, choices=["onnx", "pytorch"], default="onnx",
                       help="Model type")
    parser.add_argument("--scaler", type=str, default="scaler_params.json",
                       help="Path to scaler parameters")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for inference")
    
    # Validation options
    parser.add_argument("--dataset", type=str,
                       help="Single dataset for validation")
    parser.add_argument("--datasets", type=str, nargs="+",
                       help="Multiple datasets for validation")
    parser.add_argument("--cross-validation", action="store_true",
                       help="Perform k-fold cross-validation")
    parser.add_argument("--threshold-optimization", action="store_true",
                       help="Perform threshold optimization")
    parser.add_argument("--multi-dataset", action="store_true",
                       help="Validate across multiple datasets")
    
    # Output
    parser.add_argument("--output", type=str, default="validation_results.json",
                       help="Output file for validation results")
    
    args = parser.parse_args()
    config = Config(args)
    
    # Validate inputs
    if not any([args.cross_validation, args.threshold_optimization, args.multi_dataset]):
        print("❌ No validation method selected. Use at least one of:")
        print("   --cross-validation")
        print("   --threshold-optimization") 
        print("   --multi-dataset")
        return
    
    if (args.cross_validation or args.threshold_optimization) and not args.dataset:
        print("❌ --dataset required for cross-validation or threshold optimization")
        return
    
    if args.multi_dataset and not args.datasets:
        print("❌ --datasets required for multi-dataset validation")
        return
    
    # Run validation
    validator = NIDSValidator(config)
    validator.validate(
        dataset_path=args.dataset,
        dataset_paths=args.datasets,
        do_cross_validation=args.cross_validation,
        do_threshold_optimization=args.threshold_optimization,
        do_multi_dataset=args.multi_dataset
    )


if __name__ == "__main__":
    main()
