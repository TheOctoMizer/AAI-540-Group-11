# UEBA Model Deployment & Endpoint Creation on SageMaker

## Overview

This notebook handles the **end-to-end deployment** of the UEBA (User and Entity Behavior Analytics) CNN-GRU model on Amazon SageMaker. It covers infrastructure setup, data preparation for monitoring, endpoint creation, and integration with SageMaker Model Monitor and Clarify for bias and drift detection.

The project focuses on detecting **insider threats** by learning normal user behavior from logon and web activity, then flagging significant deviations as anomalies.

**Notebook Title:**  
**UEBA Model Deployment and Endpoint Creation**

---

## Key Features

- Complete SageMaker session and role initialization
- Memory-efficient data loading and daily aggregation of CERT dataset
- Heuristic anomaly labeling and stratified train/validation split
- Feature standardization with saved scaler artifact
- S3 data inspection and validation
- Data integration for bias analysis (device activity facet creation)
- Full environment validation for TensorFlow Serving deployment
- Preparation for Model Monitor, Clarify, and endpoint creation

---

## Notebook Structure

### Section 1: Initialisation, Infrastructure & Environment Setup
- Imports and dependencies (boto3, sagemaker, TensorFlow, scikit-learn, etc.)
- SageMaker Session initialization
- Environment configuration and library setup
- Retrieval of execution role and AWS region
- SageMaker SDK version validation
- Data loading, aggregation, and preparation
- Baseline data inspection from S3
- Data integration for bias analysis
- Device activity facet creation and distribution analysis

---

## Prerequisites

- SageMaker Studio or Jupyter Notebook with `conda_tensorflow2_p310` kernel
- Required packages:
  ```bash
  pip install boto3 sagemaker pandas numpy scikit-learn joblib tensorflow