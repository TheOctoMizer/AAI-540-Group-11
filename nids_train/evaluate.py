"""
NIDS Evaluation Script - Comprehensive model testing and analysis.

This script handles:
1. Model testing on benign and attack datasets
2. Comprehensive metrics calculation
3. Attack-type specific analysis
4. Performance comparison across thresholds
5. Result visualization and export
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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort

from preprocessing import NIDSPreprocessor
from train import Autoencoder


class NIDSEvaluator:
    """Comprehensive NIDS model evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = NIDSPreprocessor(config.scaler_path)
        self.model = None
        self.thresholds = self._load_thresholds()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def _load_thresholds(self) -> Dict[str, float]:
        """Load detection thresholds from file."""
        try:
            with open(self.config.threshold_path, "r") as f:
                thresholds = json.load(f)
            print(f"✅ Loaded thresholds from {self.config.threshold_path}")
            return thresholds
        except FileNotFoundError:
            print(f"⚠️  Threshold file not found: {self.config.threshold_path}")
            print("Will calculate thresholds from validation data if needed")
            return {}
    
    def load_model(self, input_dim: int):
        """Load the model (PyTorch or ONNX)."""
        if self.config.model_type.lower() == "pytorch":
            self.model = Autoencoder(input_dim)
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ PyTorch model loaded from {self.config.model_path}")
        elif self.config.model_type.lower() == "onnx":
            self.model = ort.InferenceSession(
                self.config.model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            print(f"✅ ONNX model loaded from {self.config.model_path}")
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def predict_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction errors for all samples."""
        if self.config.model_type.lower() == "pytorch":
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)
                mse = nn.functional.mse_loss(outputs, X_tensor, reduction='none')
                return mse.mean(dim=1).cpu().numpy()
        else:  # ONNX
            batch_size = self.config.batch_size
            errors = []
            
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                outputs = self.model.run([self.output_name], {self.input_name: batch})[0]
                mse = np.mean((outputs - batch) ** 2, axis=1)
                errors.extend(mse)
            
            return np.array(errors)
    
    def evaluate_dataset(self, data_path: str, threshold_percentile: float = 99.0) -> Dict[str, Any]:
        """Evaluate model on a dataset with comprehensive metrics."""
        print(f"\n🔍 Evaluating on: {Path(data_path).name}")
        
        # Load and preprocess data
        X, labels = self.preprocessor.preprocess_data(data_path)
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model(X.shape[1])
        
        # Get reconstruction errors
        start_time = time.time()
        errors = self.predict_reconstruction_errors(X)
        inference_time = time.time() - start_time
        
        # Determine threshold
        if f"p{threshold_percentile}" in self.thresholds:
            threshold = self.thresholds[f"p{threshold_percentile}"]
            print(f"Using saved {threshold_percentile}th percentile threshold: {threshold:.6f}")
        else:
            # For mixed datasets, calculate 95th percentile as default
            threshold = np.percentile(errors, 95)
            print(f"Using calculated 95th percentile threshold: {threshold:.6f}")
        
        # Make predictions
        predictions = errors > threshold
        
        # Convert labels to binary (benign vs attack)
        binary_labels = np.array([1 if label != "BENIGN" else 0 for label in labels])
        
        # Calculate comprehensive metrics
        tn, fp, fn, tp = confusion_matrix(binary_labels, predictions).ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # Error rates
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            "dataset": Path(data_path).name,
            "threshold": threshold,
            "threshold_percentile": threshold_percentile,
            "total_samples": len(X),
            "benign_samples": np.sum(binary_labels == 0),
            "attack_samples": np.sum(binary_labels == 1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "detection_rate": detection_rate,
            "mean_reconstruction_error": float(errors.mean()),
            "std_reconstruction_error": float(errors.std()),
            "max_reconstruction_error": float(errors.max()),
            "inference_time_seconds": inference_time,
            "samples_per_second": len(X) / inference_time
        }
        
        # Add AUC if we have both classes
        if len(np.unique(binary_labels)) > 1:
            try:
                metrics["auc_roc"] = roc_auc_score(binary_labels, errors)
            except ValueError:
                metrics["auc_roc"] = 0.0
        
        return metrics, errors, binary_labels, labels
    
    def analyze_attack_types(self, errors: np.ndarray, predictions: np.ndarray, 
                           labels: np.ndarray) -> Dict[str, Any]:
        """Analyze performance by attack type."""
        attack_analysis = {}
        
        # Get unique attack types
        unique_attacks = np.unique(labels)
        
        for attack_type in unique_attacks:
            attack_mask = labels == attack_type
            attack_errors = errors[attack_mask]
            attack_predictions = predictions[attack_mask]
            
            # Calculate metrics for this attack type
            if attack_type == "BENIGN":
                # For benign, we want false positives (incorrectly flagged as attacks)
                detections = np.sum(attack_predictions)  # False positives
                total = len(attack_predictions)
                false_positive_rate = detections / total if total > 0 else 0
                
                attack_analysis[attack_type] = {
                    "samples": int(total),
                    "false_positives": int(detections),
                    "false_positive_rate": false_positive_rate,
                    "mean_error": float(np.mean(attack_errors)),
                    "std_error": float(np.std(attack_errors)),
                    "max_error": float(np.max(attack_errors))
                }
            else:
                # For attacks, we want true positives (correctly detected)
                detections = np.sum(attack_predictions)  # True positives
                total = len(attack_predictions)
                detection_rate = detections / total if total > 0 else 0
                
                attack_analysis[attack_type] = {
                    "samples": int(total),
                    "true_positives": int(detections),
                    "detection_rate": detection_rate,
                    "mean_error": float(np.mean(attack_errors)),
                    "std_error": float(np.std(attack_errors)),
                    "max_error": float(np.max(attack_errors))
                }
        
        return attack_analysis
    
    def threshold_sweep_analysis(self, data_path: str) -> Dict[str, Any]:
        """Analyze performance across different threshold percentiles."""
        print(f"\n📈 Performing threshold sweep analysis...")
        
        X, labels = self.preprocessor.preprocess_data(data_path)
        binary_labels = np.array([1 if label != "BENIGN" else 0 for label in labels])
        
        if self.model is None:
            self.load_model(X.shape[1])
        
        errors = self.predict_reconstruction_errors(X)
        
        # Test different threshold percentiles
        percentiles = [90, 95, 99, 99.5, 99.9]
        sweep_results = {}
        
        for percentile in percentiles:
            threshold = np.percentile(errors, percentile)
            predictions = errors > threshold
            
            tn, fp, fn, tp = confusion_matrix(binary_labels, predictions).ravel()
            
            sweep_results[f"p{percentile}"] = {
                "threshold": float(threshold),
                "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
                "precision": float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                "recall": float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                "f1_score": float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
                "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0),
                "detection_rate": float(tp / (tp + fn) if (tp + fn) > 0 else 0)
            }
        
        return sweep_results
    
    def print_comprehensive_report(self, metrics: Dict[str, Any], 
                                attack_analysis: Dict[str, Any] = None,
                                sweep_results: Dict[str, Any] = None):
        """Print comprehensive evaluation report."""
        print("\n" + "="*70)
        print(f"🚨 NIDS EVALUATION REPORT: {metrics['dataset']}")
        print("="*70)
        
        # Dataset statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Samples:     {metrics['total_samples']:,}")
        print(f"   Benign Samples:    {metrics['benign_samples']:,}")
        print(f"   Attack Samples:    {metrics['attack_samples']:,}")
        
        # Detection performance
        print(f"\n🎯 Detection Performance:")
        print(f"   Threshold:         {metrics['threshold']:.6f} ({metrics['threshold_percentile']}th percentile)")
        print(f"   Accuracy:          {metrics['accuracy']:.4f}")
        print(f"   Precision:         {metrics['precision']:.4f}")
        print(f"   Recall (DR):       {metrics['recall']:.4f}")
        print(f"   F1-Score:          {metrics['f1_score']:.4f}")
        
        if 'auc_roc' in metrics:
            print(f"   AUC-ROC:           {metrics['auc_roc']:.4f}")
        
        # Performance metrics
        print(f"\n⚡ Performance:")
        print(f"   Inference Time:    {metrics['inference_time_seconds']:.2f}s")
        print(f"   Samples/sec:       {metrics['samples_per_second']:.0f}")
        
        # Confusion matrix
        print(f"\n🔍 Confusion Matrix:")
        print(f"   True Positives:    {metrics['true_positives']:,}")
        print(f"   False Positives:   {metrics['false_positives']:,}")
        print(f"   True Negatives:    {metrics['true_negatives']:,}")
        print(f"   False Negatives:   {metrics['false_negatives']:,}")
        
        # Error rates
        print(f"\n📈 Error Rates:")
        print(f"   False Positive Rate:  {metrics['false_positive_rate']:.4f}")
        print(f"   False Negative Rate:  {metrics['false_negative_rate']:.4f}")
        print(f"   Detection Rate:       {metrics['detection_rate']:.4f}")
        
        # Reconstruction error stats
        print(f"\n🔧 Reconstruction Error Stats:")
        print(f"   Mean:               {metrics['mean_reconstruction_error']:.6f}")
        print(f"   Std:                {metrics['std_reconstruction_error']:.6f}")
        print(f"   Max:                {metrics['max_reconstruction_error']:.6f}")
        
        # Attack type breakdown
        if attack_analysis:
            print(f"\n🎭 Attack Type Breakdown:")
            for attack_type, results in attack_analysis.items():
                if attack_type == "BENIGN":
                    print(f"   {attack_type}:")
                    print(f"     Samples: {results['samples']:,}")
                    print(f"     False Positives: {results['false_positives']:,} ({results['false_positive_rate']*100:.1f}%)")
                else:
                    print(f"   {attack_type}:")
                    print(f"     Samples: {results['samples']:,}")
                    print(f"     Detected: {results['true_positives']:,} ({results['detection_rate']*100:.1f}%)")
                print(f"     Mean Error: {results['mean_error']:.6f}")
        
        # Threshold sweep
        if sweep_results:
            print(f"\n📊 Threshold Sweep Analysis:")
            print(f"   Percentile | Threshold | Accuracy | Precision | Recall | F1-Score")
            print(f"   ----------|-----------|----------|-----------|--------|----------")
            for percentile, results in sweep_results.items():
                print(f"   {percentile:10s} | {results['threshold']:9.6f} | "
                      f"{results['accuracy']:8.4f} | {results['precision']:9.4f} | "
                      f"{results['recall']:6.4f} | {results['f1_score']:8.4f}")
        
        print("="*70)
    
    def save_results(self, metrics: Dict[str, Any], attack_analysis: Dict[str, Any] = None,
                    sweep_results: Dict[str, Any] = None):
        """Save comprehensive results to file."""
        results = {
            "evaluation_config": {
                "model_path": self.config.model_path,
                "model_type": self.config.model_type,
                "scaler_path": self.config.scaler_path,
                "threshold_path": self.config.threshold_path,
                "batch_size": self.config.batch_size
            },
            "metrics": metrics
        }
        
        if attack_analysis:
            results["attack_analysis"] = attack_analysis
        
        if sweep_results:
            results["threshold_sweep"] = sweep_results
        
        with open(self.config.output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {self.config.output_path}")
    
    def evaluate(self, data_path: str, threshold_percentile: float = 99.0, 
                do_attack_analysis: bool = True, do_threshold_sweep: bool = False):
        """Main evaluation pipeline."""
        print("🚀 Starting NIDS Evaluation Pipeline")
        print("=" * 50)
        
        # Load preprocessor
        self.preprocessor.load_scaler()
        
        # Main evaluation
        metrics, errors, binary_labels, labels = self.evaluate_dataset(data_path, threshold_percentile)
        
        # Attack type analysis
        attack_analysis = None
        if do_attack_analysis and metrics['attack_samples'] > 0:
            predictions = errors > metrics['threshold']
            attack_analysis = self.analyze_attack_types(errors, predictions, labels)
        
        # Threshold sweep
        sweep_results = None
        if do_threshold_sweep:
            sweep_results = self.threshold_sweep_analysis(data_path)
        
        # Print report
        self.print_comprehensive_report(metrics, attack_analysis, sweep_results)
        
        # Save results
        self.save_results(metrics, attack_analysis, sweep_results)
        
        return metrics, attack_analysis, sweep_results


class Config:
    """Evaluation configuration."""
    
    def __init__(self, args):
        self.model_path = args.model
        self.model_type = args.type
        self.scaler_path = args.scaler
        self.threshold_path = args.threshold
        self.batch_size = args.batch_size
        self.output_path = args.output


def main():
    parser = argparse.ArgumentParser(description="NIDS Evaluation Pipeline")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="nids_autoencoder.onnx",
                       help="Path to model file (ONNX or PyTorch)")
    parser.add_argument("--type", type=str, choices=["onnx", "pytorch"], default="onnx",
                       help="Model type")
    parser.add_argument("--scaler", type=str, default="scaler_params.json",
                       help="Path to scaler parameters")
    parser.add_argument("--threshold", type=str, default="detection_threshold.json",
                       help="Path to detection thresholds")
    
    # Evaluation arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--threshold-percentile", type=float, default=99.0,
                       help="Threshold percentile to use")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for inference")
    
    # Analysis options
    parser.add_argument("--no-attack-analysis", action="store_true",
                       help="Skip attack type analysis")
    parser.add_argument("--threshold-sweep", action="store_true",
                       help="Perform threshold sweep analysis")
    
    # Output
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    config = Config(args)
    
    # Validate files exist
    required_files = [config.model_path, config.scaler_path, args.data]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            return
    
    # Run evaluation
    evaluator = NIDSEvaluator(config)
    evaluator.evaluate(
        data_path=args.data,
        threshold_percentile=args.threshold_percentile,
        do_attack_analysis=not args.no_attack_analysis,
        do_threshold_sweep=args.threshold_sweep
    )


if __name__ == "__main__":
    main()
