import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from classifier.train import train_xgboost_classifier


def main():
    """Entry point for XGBoost training."""
    parser = argparse.ArgumentParser(description="Train XGBoost classifier for NIDS")
    parser.add_argument("--data-dir", type=str, default="./nids_dataset", 
                       help="Directory containing the dataset")
    parser.add_argument("--dataset", type=str, default="Wednesday-workingHours.pcap_ISCX.csv",
                       help="Dataset filename")
    parser.add_argument("--output-dir", type=str, default="./model",
                       help="Directory to save the trained model")
    parser.add_argument("--autoencoder-model", type=str, default=None,
                       help="Path to autoencoder.pt weights (if provided, XGBoost trains on encoded features)")
    parser.add_argument("--latent-dim", type=int, default=8,
                       help="Autoencoder latent dimension (default: 8)")
    parser.add_argument("--no-onnx", action="store_true",
                       help="Skip ONNX export and consistency evaluation")
    parser.add_argument("--no-consistency", action="store_true",
                       help="Skip XGBoost vs ONNX consistency evaluation")
    args = parser.parse_args()

    print("Starting XGBoost classifier training...")
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Autoencoder model: {args.autoencoder_model or 'None (raw features)'}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Export ONNX: {not args.no_onnx}")
    print(f"Evaluate consistency: {not args.no_consistency}")
    print("-" * 50)

    # Train the model
    model, encoder, accuracy, onnx_consistency = train_xgboost_classifier(
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_dir=args.output_dir,
        autoencoder_path=args.autoencoder_model,
        latent_dim=args.latent_dim,
        export_onnx=not args.no_onnx,
        evaluate_consistency=not args.no_consistency,
    )

    print("-" * 50)
    print(f"Training completed successfully!")
    print(f"Test accuracy: {accuracy:.4f}")
    
    if onnx_consistency:
        print(f"ONNX consistency - Max diff: {onnx_consistency['max_difference']:.6e}")
        print(f"ONNX consistency - Mean diff: {onnx_consistency['mean_difference']:.6e}")


if __name__ == "__main__":
    main()