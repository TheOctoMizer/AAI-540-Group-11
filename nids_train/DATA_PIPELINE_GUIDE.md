# Data Pipeline User Guide

## Overview

The `data_pipeline.py` script is a comprehensive data processing pipeline for the AI NIDS project. It processes all 8 CIC-IDS2017 datasets and prepares them for AWS deployment (S3, Athena, SageMaker).

## Features

1. **Data Cataloging** - Generates Athena-compatible metadata catalog
2. **EDA** - Statistical analysis + visualization plots (class distribution, feature correlations)
3. **Feature Engineering** - Consistent preprocessing with MinMaxScaler
4. **Data Splitting** - Stratified 40% train / 10% test / 10% val / 40% production
5. **Feature Store** - Outputs in Parquet (efficient) and CSV (human-readable) formats

## Installation

The required dependencies have been added to the project:

```bash
# Dependencies installed:
# - pyarrow (for Parquet file support)
# - matplotlib (for visualizations)
# - seaborn (for statistical plots)
```

## Usage

### Run Complete Pipeline

Process all 8 datasets through all 5 phases:

```bash
cd nids_train
python data_pipeline.py --data-dir ./nids_dataset --output-dir ./output
```

### Run Specific Phases

```bash
# Data cataloging only
python data_pipeline.py --phase catalog

# EDA only
python data_pipeline.py --phase eda

# Feature engineering only
python data_pipeline.py --phase features

# Data splitting only
python data_pipeline.py --phase split

# Feature store preparation only
python data_pipeline.py --phase store
```

### Custom Split Ratios

```bash
# Example: 50% train, 20% test, 10% val, 20% production
python data_pipeline.py \
  --train-ratio 0.5 \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --prod-ratio 0.2
```

### Skip Visualizations (Faster)

```bash
python data_pipeline.py --no-plots
```

## Output Structure

After running the pipeline, you'll have:

```
output/
├── catalog/
│   └── data_catalog.json              # Athena table metadata
├── eda/
│   ├── eda_report.json                # Statistical analysis
│   ├── class_distribution.png         # Attack type distribution
│   ├── feature_statistics.png         # Top 20 features histograms
│   └── feature_correlation.png        # Correlation matrix
├── features/
│   ├── train/
│   │   ├── features.parquet          # Training features (Parquet)
│   │   ├── features.csv              # Training features (CSV)
│   │   ├── labels.csv                # Training labels
│   │   └── metadata.json             # Split metadata
│   ├── test/
│   │   ├── features.parquet
│   │   ├── features.csv
│   │   ├── labels.csv
│   │   └── metadata.json
│   ├── val/
│   │   ├── features.parquet
│   │   ├── features.csv
│   │   ├── labels.csv
│   │   └── metadata.json
│   └── production/
│       ├── features.parquet
│       ├── features.csv
│       ├── labels.csv
│       └── metadata.json
└── scaler_params.json                 # Feature scaling parameters
```

## AWS Integration Guide

### 1. Upload to S3 Datalake

```bash
# Upload raw datasets
aws s3 cp nids_dataset/ s3://your-bucket/raw-data/ --recursive

# Upload processed features
aws s3 cp output/features/ s3://your-bucket/processed-features/ --recursive

# Upload catalog
aws s3 cp output/catalog/data_catalog.json s3://your-bucket/catalog/
```

### 2. Create Athena Tables

Use the `data_catalog.json` to create Athena tables:

```sql
-- Example for training data
CREATE EXTERNAL TABLE nids_train_features (
  `Destination Port` DOUBLE,
  `Flow Duration` DOUBLE,
  -- ... (add all 77 features from catalog)
)
STORED AS PARQUET
LOCATION 's3://your-bucket/processed-features/train/';

CREATE EXTERNAL TABLE nids_train_labels (
  label STRING
)
STORED AS PARQUET
LOCATION 's3://your-bucket/processed-features/train/';
```

### 3. SageMaker Notebook - EDA

```python
import pandas as pd
import json

# Load EDA report
with open('s3://your-bucket/eda/eda_report.json') as f:
    eda = json.load(f)

# Load features for analysis
train_df = pd.read_parquet('s3://your-bucket/processed-features/train/features.parquet')

# Your analysis here...
```

### 4. Feature Store Setup

```python
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Create feature group
feature_group = FeatureGroup(
    name="nids-features",
    sagemaker_session=sagemaker.Session()
)

# Load features
train_df = pd.read_parquet('output/features/train/features.parquet')
labels_df = pd.read_csv('output/features/train/labels.csv')

# Combine and ingest
data = pd.concat([train_df, labels_df], axis=1)
data['event_time'] = pd.Timestamp.now().isoformat()

feature_group.ingest(data_frame=data, max_workers=3, wait=True)
```

## Data Catalog Schema

The `data_catalog.json` contains:

```json
{
  "created_at": "2026-02-04T15:47:08",
  "source": "CIC-IDS2017",
  "num_datasets": 8,
  "total_samples": 2830743,
  "total_size_mb": 884.36,
  "num_features": 77,
  "attack_types": [
    "BENIGN", "DoS", "DDoS", "PortScan", "FTP-Patator",
    "SSH-Patator", "Web Attack", "Infiltration", "Bot"
  ],
  "feature_names": [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    ...
  ],
  "datasets": [
    {
      "filename": "Monday-WorkingHours.pcap_ISCX.csv",
      "size_mb": 177.0,
      "num_samples": 529918,
      "class_distribution": {"BENIGN": 529918}
    },
    ...
  ]
}
```

## EDA Report Schema

The `eda_report.json` contains:

```json
{
  "total_samples": 2830743,
  "num_features": 77,
  "class_distribution": {
    "BENIGN": 2273097,
    "DoS": 252672,
    ...
  },
  "feature_statistics": {
    "Destination Port": {
      "mean": 12345.67,
      "std": 8901.23,
      "min": 0,
      "max": 65535,
      "p50": 443,
      "p95": 50000,
      "p99": 60000
    },
    ...
  },
  "missing_values": {...},
  "infinite_values": {...}
}
```

## Feature Store Metadata

Each split's `metadata.json` includes:

```json
{
  "split": "train",
  "num_samples": 1132297,
  "num_features": 77,
  "feature_names": ["Destination Port", ...],
  "class_distribution": {
    "BENIGN": 909239,
    "DoS": 101069,
    ...
  },
  "preprocessing": {
    "scaler": "MinMaxScaler",
    "scaler_params_file": "../scaler_params.json"
  },
  "created_at": "2026-02-04T15:47:08"
}
```

## Tips

1. **Large Dataset Processing**: The pipeline processes ~884 MB of data. Expect 5-10 minutes runtime.
2. **Memory Requirements**: Ensure at least 8 GB RAM available.
3. **Parquet vs CSV**: Use Parquet for analytics (faster, smaller), CSV for manual inspection.
4. **Scaler Consistency**: Always use `scaler_params.json` for production inference.
5. **Stratified Splits**: Class distribution is preserved across all splits.

## Troubleshooting

### Out of Memory

```bash
# Process fewer datasets or increase swap
# Or run phases separately to reduce memory footprint
python data_pipeline.py --phase features
```

### Missing Datasets

The script will skip missing files and continue. Check console output for warnings.

### Visualization Errors

```bash
# Skip plots if headless environment
python data_pipeline.py --no-plots
```

## Next Steps

After running the pipeline:

1. ✅ Review `eda_report.json` and visualizations
2. ✅ Verify split ratios in feature store metadata
3. ✅ Upload to S3 datalake
4. ✅ Create Athena tables using catalog schema
5. ✅ Run exploratory analysis in SageMaker notebook
6. ✅ Ingest features into Feature Store
7. ✅ Use training data for model training
8. ✅ Reserve production split for final evaluation

## Command Reference

```bash
# Full pipeline with all datasets
python data_pipeline.py

# Custom output location
python data_pipeline.py --output-dir /path/to/output

# Different split ratios
python data_pipeline.py --train-ratio 0.5 --test-ratio 0.2 --val-ratio 0.1 --prod-ratio 0.2

# Fast run (no plots)
python data_pipeline.py --no-plots

# Just cataloging
python data_pipeline.py --phase catalog

# Just EDA
python data_pipeline.py --phase eda

# Custom seed for reproducibility
python data_pipeline.py --seed 123
```
