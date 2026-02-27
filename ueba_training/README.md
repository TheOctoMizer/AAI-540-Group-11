# UEBA Preprocessing Pipeline for CNN-GRU Model

## Overview

This notebook implements a complete **User and Entity Behavior Analytics (UEBA)** preprocessing pipeline for the CERT insider threat dataset. It prepares large-scale behavioral data for training a **Hybrid CNN-GRU** model in SageMaker.

The pipeline handles memory-efficient processing of ~955k records, feature engineering, scaling, stratified splitting, and binary serialization — optimized for production-grade security analytics.

**Suggested Notebook Title:**  
**UEBA Data Preprocessing Pipeline**

---

## Key Features

- **Memory-Efficient Chunking** — Processes data in 100,000-row chunks to avoid session crashes
- **Behavioral Feature Engineering** — Label encoding + StandardScaler for numerical and categorical features
- **Stratified Train/Validation Split** — 80/20 split preserving rare anomaly class distribution
- **SageMaker Integration** — Automatic S3 upload/download and session management
- **Benchmark Compatibility** — Maps UEBA features to a Breast Cancer-like diagnostic schema for cross-domain validation
- **Fast Loading** — Saves data as `.npy` files for quick SageMaker training

---

## Notebook Structure

### 1. UEBA Preprocessing Pipeline
- Chunked reading of raw data
- Label encoding and scaling
- Stratified 80/20 split
- Save as NumPy binary files

### 2. SageMaker Session & S3 Setup
- Initialize SageMaker session and execution role
- Define bucket paths for model artifacts
- Load and validate training data

### 3. UEBA Behavioral Feature Listing
- Temporal averaging across 50-day windows
- Human-readable feature summary with diagnosis labels
- Tab-separated output for audit logs

### 4. UEBA Data Preparation (Benchmark Mapping)
- Flatten 3D temporal data to 2D
- Map UEBA features to clinical-style columns (`radius_mean`, `texture_mean`, etc.)
- Generate synthetic features to match 32-column benchmark format
- Boolean label transformation (`M` = Anomaly, `B` = Normal)

### 5. Model Configuration Summary
- Displays CNN-GRU architecture details
- Hyperparameters table (filters, GRU units, optimizer, epochs, etc.)

---

## Prerequisites

- Python 3.10+
- Required packages:
  ```bash
  pip install pandas numpy scikit-learn boto3 sagemaker