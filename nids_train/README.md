# Network Intrusion Detection System (NIDS) - Consolidated Version

A streamlined autoencoder-based Network Intrusion Detection System with 3 main scripts: **Train**, **Evaluate**, and **Validate**.

## Overview

This NIDS uses an autoencoder neural network trained on benign network traffic to detect anomalies and attacks. The consolidated structure provides a clean, production-ready pipeline with comprehensive evaluation and validation capabilities.

### Key Features
- **3-Script Architecture**: Train, Evaluate, Validate - clean separation of concerns
- **Autoencoder Architecture**: 70+ features → 64 → 32 → 8 → 32 → 64 → 70+ features
- **Multi-format Support**: PyTorch training with ONNX export for production
- **Comprehensive Evaluation**: Detailed metrics, attack-type analysis, threshold optimization
- **Production Ready**: All asset generation integrated into training script
- **Robust Validation**: Cross-validation, threshold optimization, multi-dataset testing

## Consolidated Project Structure

```
nids_train/
├── Core Scripts (NEW)
│   ├── train.py                    # Training + asset export (consolidated)
│   ├── evaluate.py                 # Comprehensive testing (consolidated)
│   └── validate.py                 # Validation & threshold analysis (NEW)
│
├── Supporting Modules
│   ├── preprocessing.py             # Shared preprocessing utilities
│   └── README_consolidated.md       # This file
│
├── Dataset  
│   └── nids_dataset/               # CIC-IDS2017 dataset files
│
└── Legacy Scripts (deprecated)
    ├── train_nids.py               # Old training script
    ├── test_*.py                   # Old testing scripts
    ├── export_*.py                 # Old export scripts
    └── evaluate_nids.py            # Old evaluation script
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install torch pandas scikit-learn onnxruntime matplotlib seaborn

# Download CIC-IDS2017 dataset to nids_dataset/
# Required files:
# - Monday-WorkingHours.pcap_ISCX.csv (benign traffic)
# - Wednesday-workingHours.pcap_ISCX.csv (attacks)
```

### 2. Train Model (with automatic asset export)

```bash
# Basic training
python train.py

# Custom training parameters
python train.py --epochs 20 --batch-size 2048 --learning-rate 1e-4

# Custom output paths
python train.py --onnx-output models/nids_v2.onnx --scaler-output models/scaler_v2.json
```

**Generated Assets:**
- `nids_autoencoder.pt` - PyTorch model
- `nids_autoencoder.onnx` - ONNX model  
- `scaler_params.json` - Preprocessing parameters
- `detection_threshold.json` - Detection thresholds
- `test_samples.json` - Test samples for validation
- `training_metadata.json` - Training configuration and history

### 3. Evaluate Model

```bash
# Basic evaluation
python evaluate.py --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv

# Comprehensive evaluation with threshold sweep
python evaluate.py --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv --threshold-sweep

# Evaluate PyTorch model
python evaluate.py --model nids_autoencoder.pt --type pytorch --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv

# Custom threshold percentile
python evaluate.py --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv --threshold-percentile 95.0
```

### 4. Validate Model

```bash
# Cross-validation (requires benign dataset)
python validate.py --dataset ./nids_dataset/Monday-WorkingHours.pcap_ISCX.csv --cross-validation

# Threshold optimization
python validate.py --dataset ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv --threshold-optimization

# Multi-dataset validation
python validate.py --datasets ./nids_dataset/Tuesday-WorkingHours.pcap_ISCX.csv ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv --multi-dataset

# Comprehensive validation (all methods)
python validate.py --dataset ./nids_dataset/Monday-WorkingHours.pcap_ISCX.csv --cross-validation --threshold-optimization
```

## Detailed Usage

### Training Script (`train.py`)

**Features:**
- Complete training pipeline with validation
- Automatic asset export (PyTorch, ONNX, scaler, thresholds)
- Training history tracking
- Configurable hyperparameters
- Test sample generation

**Command Options:**
```bash
python train.py [OPTIONS]

Data Options:
  --dataset PATH                Training dataset path
  --val-split FLOAT             Validation split ratio (default: 0.2)

Training Options:
  --batch-size INT              Batch size (default: 1024)
  --epochs INT                  Training epochs (default: 10)
  --learning-rate FLOAT         Learning rate (default: 1e-3)

Output Options:
  --pytorch-output PATH         PyTorch model output
  --onnx-output PATH            ONNX model output
  --scaler-output PATH          Scaler parameters output
  --threshold-output PATH       Detection thresholds output
  --test-samples-output PATH    Test samples output
  --metadata-output PATH        Training metadata output
```

**Example:**
```bash
python train.py \
  --dataset ./nids_dataset/Monday-WorkingHours.pcap_ISCX.csv \
  --epochs 15 \
  --batch-size 2048 \
  --learning-rate 5e-4 \
  --onnx-output models/nids_prod.onnx \
  --threshold-output models/thresholds.json
```

### Evaluation Script (`evaluate.py`)

**Features:**
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Attack-type specific analysis
- Threshold sweep analysis
- Performance benchmarking
- Detailed result export

**Command Options:**
```bash
python evaluate.py [OPTIONS]

Model Options:
  --model PATH                  Model file path
  --type {onnx,pytorch}        Model type (default: onnx)
  --scaler PATH                 Scaler parameters path
  --threshold PATH              Threshold file path

Evaluation Options:
  --data PATH                   Test dataset path (required)
  --threshold-percentile FLOAT Threshold percentile (default: 99.0)
  --batch-size INT              Batch size (default: 1024)

Analysis Options:
  --no-attack-analysis         Skip attack type analysis
  --threshold-sweep            Perform threshold sweep analysis

Output:
  --output PATH                 Results output file
```

**Example:**
```bash
python evaluate.py \
  --model nids_autoencoder.onnx \
  --data ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv \
  --threshold-sweep \
  --output results/wednesday_evaluation.json
```

### Validation Script (`validate.py`)

**Features:**
- K-fold cross-validation
- Threshold optimization (percentile, Youden's J, max F1)
- Multi-dataset validation
- Statistical significance testing
- Robustness analysis

**Command Options:**
```bash
python validate.py [OPTIONS]

Model Options:
  --model PATH                  Model file path
  --type {onnx,pytorch}        Model type (default: onnx)
  --scaler PATH                 Scaler parameters path
  --batch-size INT              Batch size (default: 1024)

Validation Options:
  --dataset PATH                Single dataset for validation
  --datasets PATH [PATH ...]    Multiple datasets for validation
  --cross-validation            Perform k-fold cross-validation
  --threshold-optimization      Perform threshold optimization
  --multi-dataset              Validate across multiple datasets

Output:
  --output PATH                 Validation results output file
```

**Example:**
```bash
# Comprehensive validation
python validate.py \
  --dataset ./nids_dataset/Monday-WorkingHours.pcap_ISCX.csv \
  --cross-validation \
  --threshold-optimization \
  --output results/comprehensive_validation.json

# Multi-dataset robustness check
python validate.py \
  --datasets ./nids_dataset/Tuesday-WorkingHours.pcap_ISCX.csv \
            ./nids_dataset/Wednesday-workingHours.pcap_ISCX.csv \
            ./nids_dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv \
  --multi-dataset \
  --output results/robustness_check.json
```

## Evaluation Metrics

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

### Threshold Optimization Methods
1. **Percentile-based**: 90th, 95th, 99th, 99.5th, 99.9th percentiles
2. **Youden's J Statistic**: Maximizes sensitivity + specificity - 1
3. **Maximum F1-Score**: Optimizes for F1-score

## Production Deployment

### Asset Generation
The training script automatically generates all production assets:

```bash
python train.py
```

**Required Production Files:**
- `nids_autoencoder.onnx` - Model weights
- `scaler_params.json` - Feature scaling parameters
- `detection_threshold.json` - Detection thresholds

### Integration Example (Rust)
```rust
// Load assets
let scaler_params = load_scaler("scaler_params.json")?;
let thresholds = load_thresholds("detection_threshold.json")?;
let session = ort::Session::new(&env, "nids_autoencoder.onnx")?;

// Process network flow
let scaled_features = scale_features(flow, &scaler_params);
let reconstruction = session.run(scaled_features)?;
let error = calculate_mse(&reconstruction, &scaled_features);

// Detect intrusion
let is_attack = error > thresholds.p99;
```

## Advanced Configuration

### Custom Training Configuration
```python
# Modify training parameters in train.py or use CLI
config = {
    "batch_size": 2048,
    "epochs": 20,
    "learning_rate": 1e-4,
    "val_split": 0.15
}
```

### Threshold Selection Strategy
- **High Security**: Use 95th percentile (lower threshold, more sensitive)
- **Balanced**: Use 99th percentile (recommended)
- **Low False Positives**: Use 99.9th percentile (higher threshold)

### Cross-Validation for Robustness
```bash
# 5-fold cross-validation on benign data
python validate.py --dataset benign_data.csv --cross-validation
```

## Expected Performance

Based on CIC-IDS2017 dataset:

| Metric | Benign Data | Attack Data |
|--------|-------------|-------------|
| Accuracy | > 99% | 85-95% |
| Detection Rate | N/A | 80-95% |
| False Positive Rate | < 1% | N/A |
| F1-Score | N/A | 0.85-0.95 |

*Performance varies by attack type and threshold selection*

## Troubleshooting

### Common Issues

1. **Missing Assets**
   ```
   FileNotFoundError: scaler_params.json
   ```
   **Solution**: Run `python train.py` to generate all assets

2. **Model Architecture Mismatch**
   ```
   Error: size mismatch between model and data
   ```
   **Solution**: Ensure all scripts use the same preprocessing pipeline

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce batch size or use CPU training

### Performance Optimization

- **Training**: Increase batch size, use GPU acceleration
- **Inference**: Use ONNX Runtime, batch processing
- **Threshold**: Use validation set to optimize threshold

## Migration from Legacy Scripts

### From Old Training Scripts
```bash
# OLD
python train_nids.py
python export_scaler.py
python export_rust_assets.py

# NEW (consolidated)
python train.py
```

### From Old Testing Scripts
```bash
# OLD
python test_attacks_onnx.py
python evaluate_nids.py --data attacks.csv

# NEW (consolidated)
python evaluate.py --data attacks.csv --threshold-sweep
```

## Dataset Information

**CIC-IDS2017 Dataset** - Realistic network traffic capture:
- **Monday**: Benign traffic only (training/validation)
- **Tuesday**: FTP/SSH brute force attacks  
- **Wednesday**: DoS/DDoS attacks
- **Thursday**: Web attacks, infiltration
- **Friday**: Botnet, port scan, DDoS attacks

### Recommended Dataset Usage
- **Training**: Monday-WorkingHours.pcap_ISCX.csv (benign only)
- **Validation**: Split from Monday data
- **Testing**: Any other day's data (contains attacks)
- **Robustness**: Multiple days for cross-validation

## Contributing

1. **Code Style**: Follow PEP 8, use type hints
2. **Testing**: Add validation tests for new features
3. **Documentation**: Update README for API changes
4. **Performance**: Benchmark changes before submission

## License

This project is provided for educational and research purposes. Please cite the original CIC-IDS2017 dataset if used in publications.

## References

- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Autoencoder Anomaly Detection](https://arxiv.org/abs/1811.03259)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch](https://pytorch.org/)
