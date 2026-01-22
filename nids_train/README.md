# Network Intrusion Detection System (NIDS)

A comprehensive autoencoder-based Network Intrusion Detection System using PyTorch with ONNX export for production deployment.

## Overview

This NIDS uses an autoencoder neural network trained on benign network traffic to detect anomalies and attacks. The system learns to reconstruct normal traffic patterns, with high reconstruction errors indicating potential intrusions.

### Key Features
- **Autoencoder Architecture**: 70+ features → 64 → 32 → 8 → 32 → 64 → 70+ features
- **Multi-format Support**: PyTorch training with ONNX export for production
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, attack-type analysis
- **Production Ready**: Scaler parameter export for Rust/AWS Lambda deployment
- **Consistent Preprocessing**: Shared preprocessing pipeline ensures reproducibility

## Project Structure

```
nids_train/
├── Core Scripts
│   ├── train_nids.py              # Main training script
│   ├── preprocessing.py           # Shared preprocessing utilities
│   └── evaluate_nids.py           # Comprehensive evaluation framework
│
├── Testing Scripts  
│   ├── test_nids_torch.py         # PyTorch model testing
│   ├── test_nids_onnx.py          # ONNX model testing
│   └── test_attacks_onnx.py       # Attack detection evaluation
│
├── Production Export
│   ├── export_scaler.py           # Export scaler parameters
│   ├── export_rust_assets.py      # Export all assets for Rust
│   └── export_test_samples.py     # Export test samples
│
├── Dataset
│   └── nids_dataset/              # CIC-IDS2017 dataset files
│
└── Generated Assets
    ├── nids_autoencoder.pt        # PyTorch model
    ├── nids_autoencoder.onnx      # ONNX model  
    ├── scaler_params.json         # Scaler parameters
    └── detection_threshold.json   # Detection thresholds
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install torch pandas scikit-learn onnxruntime

# Download CIC-IDS2017 dataset to nids_dataset/
# Required files:
# - Monday-WorkingHours.pcap_ISCX.csv (benign traffic)
# - Wednesday-workingHours.pcap_ISCX.csv (attacks)
```

### 2. Train Model

```bash
python train_nids.py
```

**Output:**
- `nids_autoencoder.pt` - PyTorch model
- `nids_autoencoder.onnx` - ONNX model  
- `scaler_params.json` - Preprocessing parameters
- `detection_threshold.json` - Detection thresholds

### 3. Evaluate Model

```bash
# Test on benign traffic
python test_nids_onnx.py

# Test attack detection
python test_attacks_onnx.py

# Comprehensive evaluation with metrics
python evaluate_nids.py --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv
```

## Usage Examples

### Training with Custom Parameters

```python
# Edit train_nids.py constants
BATCH_SIZE = 2048
EPOCHS = 20
LEARNING_RATE = 1e-4
```

### Attack Detection with Custom Threshold

```bash
python test_attacks_onnx.py --threshold 95.0 --output custom_results.json
```

### Comprehensive Evaluation

```bash
# Evaluate ONNX model
python evaluate_nids.py --model nids_autoencoder.onnx --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv --type onnx

# Evaluate PyTorch model  
python evaluate_nids.py --model nids_autoencoder.pt --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv --type pytorch
```

## Model Architecture

### Autoencoder Structure
```
Input (78 features) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(8) → ReLU
                     ↓
                 Latent Space (8 dimensions)
                     ↓  
Dense(32) → ReLU → Dense(64) → ReLU → Dense(78) → Sigmoid → Output
```

### Training Details
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=1e-3)
- **Training Data**: Only benign traffic (Monday dataset)
- **Validation**: 20% split of benign data
- **Device**: MPS (Apple Silicon) or CPU fallback

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Detection Performance
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate of attack detection  
- **Recall (Detection Rate)**: Ability to detect actual attacks
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Error Analysis
- **False Positive Rate**: Benign traffic incorrectly flagged
- **False Negative Rate**: Attacks missed by the system
- **Reconstruction Error Statistics**: Mean, std, percentiles

### Attack Type Breakdown
Detailed analysis by attack category:
- DoS (Denial of Service)
- PortScan
- Web Attacks
- Infiltration
- Botnet
- DDoS

## Production Deployment

### Export Assets for Rust/AWS Lambda

```bash
# Export scaler parameters
python export_scaler.py

# Export all required assets  
python export_rust_assets.py

# Export test samples for validation
python export_test_samples.py
```

### Required Production Files
- `nids_autoencoder.onnx` - Model weights
- `scaler_params.json` - Feature scaling parameters
- `detection_threshold.json` - Detection thresholds

### Integration Example (Rust)
```rust
// Load scaler parameters
let scaler_params = load_scaler("scaler_params.json")?;

// Load ONNX model
let session = ort::Session::new(&env, "nids_autoencoder.onnx")?;

// Process network flow
let scaled_features = scale_features(flow, &scaler_params);
let reconstruction = session.run(scaled_features)?;
let error = calculate_mse(&reconstruction, &scaled_features);

// Detect intrusion
let is_attack = error > threshold;
```

## Advanced Configuration

### Custom Preprocessing
```python
from preprocessing import NIDSPreprocessor

# Initialize with custom scaler path
preprocessor = NIDSPreprocessor("custom_scaler.json")

# Fit on training data
preprocessor.fit_scaler("training_data.csv")

# Process new data
X, labels = preprocessor.preprocess_data("test_data.csv")
```

### Threshold Selection
The system automatically calculates multiple threshold percentiles:
- **95th percentile**: Lower false positives, higher false negatives
- **99th percentile**: Balanced detection (recommended)
- **99.9th percentile**: Lower false negatives, higher false positives

### Batch Processing
```python
# Process large datasets efficiently
from evaluate_nids import NIDSEvaluator

evaluator = NIDSEvaluator("nids_autoencoder.onnx")
metrics = evaluator.evaluate_dataset("large_dataset.csv")
```

## Troubleshooting

### Common Issues

1. **Model Architecture Mismatch**
   ```
   Error: size mismatch between saved model and current architecture
   ```
   **Solution**: Ensure all scripts use the same autoencoder architecture (8-dim latent space)

2. **Scaler Inconsistency**  
   ```
   Error: Feature dimension mismatch
   ```
   **Solution**: Use `export_scaler.py` to regenerate scaler parameters

3. **Missing Threshold File**
   ```
   FileNotFoundError: detection_threshold.json
   ```
   **Solution**: Run training script to generate thresholds, or use evaluate_nids.py for automatic calculation

### Performance Optimization

- **Batch Size**: Increase for faster training (memory permitting)
- **GPU Acceleration**: Use CUDA-enabled PyTorch for training
- **ONNX Optimization**: Use ONNX Runtime for optimized inference

## Dataset Information

**CIC-IDS2017 Dataset** - Realistic network traffic capture:
- **Monday**: Benign traffic only (training)
- **Tuesday**: FTP/SSH brute force attacks  
- **Wednesday**: DoS/DDoS attacks
- **Thursday**: Web attacks, infiltration
- **Friday**: Botnet, port scan, DDoS attacks

### Feature Engineering
78 network flow features including:
- Basic features (duration, protocol, packet counts)
- Flow statistics (IAT, flow bytes/packets)
- Time-related features (forward/backward IAT)
- Header statistics (TCP flags, packet size)

## Contributing

1. **Code Style**: Follow PEP 8, use type hints
2. **Testing**: Add unit tests for new features
3. **Documentation**: Update README for API changes
4. **Performance**: Benchmark changes before submission

## License

This project is provided for educational and research purposes. Please cite the original CIC-IDS2017 dataset if used in publications.

## References

- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Autoencoder Anomaly Detection](https://arxiv.org/abs/1811.03259)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch](https://pytorch.org/)
