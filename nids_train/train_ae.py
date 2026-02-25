"""
SageMaker-compatible training entry point for NIDS Autoencoder.

Responsibilities:
- Load and preprocess dataset
- Filter benign traffic
- Split data: 40/10/10/40
- Train autoencoder
- Save trained model artifact

Non-responsibilities:
- Thresholding
- ONNX export
- Inference
"""

import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from preprocessing.nids_preprocessor import NIDSPreprocessor
from models.autoencoder import Autoencoder
from data.splits import split_benign_data
from training.train import train_autoencoder
from export.onnx_export import export_onnx
from export.onnx_quantize import quantize_onnx

# -----------------------------
# Argument parsing (SageMaker-style)
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker standard paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    # Dataset
    parser.add_argument("--dataset", type=str, default="Monday-WorkingHours.pcap_ISCX.csv")

    return parser.parse_args()


# -----------------------------
# Main training routine
# -----------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve paths
    dataset_path = Path(args.data_dir) / args.dataset
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load and preprocess data (Global Benign)
    # -----------------------------
    print("Aggregating benign samples from all dataset files...")
    preprocessor = NIDSPreprocessor()
    X_benign = preprocessor.aggregate_benign_data(
        args.data_dir,
        fit_scaler=True,
    )
    
    print(f"Total global benign samples: {len(X_benign)}")

    # -----------------------------
    # Split data (40/10/10/40)
    # -----------------------------
    splits = split_benign_data(X_benign, random_state=args.random_state)

    X_train = torch.FloatTensor(splits["train"])
    X_val   = torch.FloatTensor(splits["val"])
    X_test  = torch.FloatTensor(splits["test"])
    X_prod  = splits["production"]  # intentionally unused

    print("Data split summary:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    print(f"  Prod:  {len(X_prod)}")

    # -----------------------------
    # Train model
    # -----------------------------
    model = Autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=args.latent_dim
    ).to(device)

    config = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate
    )

    model, history, val_errors = train_autoencoder(
        model=model,
        X_train=X_train,
        X_val=X_val,
        config=config,
        device=device
    )

    # Calculate additional metrics
    final_train_loss = float(history[-1]["train_loss"])
    final_val_loss = float(history[-1]["val_loss"])
    best_val_loss = float(min([h["val_loss"] for h in history]))
    total_training_time = float(history[-1]["total_time"])
    avg_epoch_time = float(sum([h["epoch_time"] for h in history]) / len(history))
    
    # Calculate improvement metrics
    initial_val_loss = float(history[0]["val_loss"])
    overall_improvement = ((initial_val_loss - final_val_loss) / initial_val_loss * 100) if initial_val_loss > 0 else 0
    
    # Validation error statistics
    val_mean = float(np.mean(val_errors))
    val_std = float(np.std(val_errors))
    val_median = float(np.median(val_errors))
    val_p95 = float(np.percentile(val_errors, 95))
    val_p99 = float(np.percentile(val_errors, 99))

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Final Train Loss:     {final_train_loss:.6f}")
    print(f"Final Val Loss:       {final_val_loss:.6f}")
    print(f"Best Val Loss:        {best_val_loss:.6f}")
    print(f"Overall Improvement:  {overall_improvement:+.2f}%")
    print(f"Total Training Time:  {total_training_time:.2f} seconds")
    print(f"Average Epoch Time:   {avg_epoch_time:.2f} seconds")
    print(f"\nValidation Error Statistics:")
    print(f"  Mean:               {val_mean:.6f}")
    print(f"  Std Dev:            {val_std:.6f}")
    print(f"  Median:             {val_median:.6f}")
    print(f"  95th Percentile:    {val_p95:.6f}")
    print(f"  99th Percentile:    {val_p99:.6f}")
    print("=" * 80)

    onnx_path = model_dir / "autoencoder.onnx"
    export_onnx(model, X_train.shape[1], onnx_path)
    print(f"ONNX FP32 model saved to: {onnx_path}")

    int8_path = model_dir / "autoencoder_int8.onnx"
    quantize_onnx(onnx_path, int8_path)
    print(f"ONNX INT8 model saved to: {int8_path}")

    # -----------------------------
    # Save model artifact
    # -----------------------------
    model_path = model_dir / "autoencoder.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")

    # -----------------------------
    # Save training summary
    # -----------------------------
    summary = {
        "input_dim": X_train.shape[1],
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "overall_improvement_percent": overall_improvement,
        "total_training_time_seconds": total_training_time,
        "avg_epoch_time_seconds": avg_epoch_time,
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val),
        "num_test_samples": len(X_test),
        "num_production_samples": len(X_prod),
        "validation_stats": {
            "mean": val_mean,
            "std": val_std,
            "median": val_median,
            "p95": val_p95,
            "p99": val_p99
        },
        "training_history": [
            {
                "epoch": int(h["epoch"]),
                "train_loss": float(h["train_loss"]),
                "val_loss": float(h["val_loss"]),
                "epoch_time": float(h["epoch_time"]),
                "total_time": float(h["total_time"])
            }
            for h in history
        ]
    }

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()