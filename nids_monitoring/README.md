# SageMaker Model Monitoring Setup & Debugging for UEBA Project

## Overview

This notebook provides a comprehensive set of tools and scripts to **set up, inspect, and debug** SageMaker Model Monitoring for the UEBA (User and Entity Behavior Analytics) project.

It covers both **Data Quality Monitoring** and **Model Quality Monitoring**, including real-time endpoint inspection, baseline generation, schedule health checks, and detailed configuration auditing.

**Notebook Title:**  
**SageMaker Model Monitoring Setup & Debugging**

---

## Key Features

- Real-time inspection of deployed SageMaker endpoints
- IAM Role and SageMaker version verification
- Data Quality Baseline Job creation
- Model Quality Monitoring Schedule automation (with Ground Truth)
- Comprehensive schedule health and configuration debugging
- S3 path validation for baselines, ground truth, and output locations
- Full raw response printing for advanced troubleshooting

---

## Notebook Structure

### 1. Real-Time Inspection of a Deployed AI Model
- Describe endpoint details (ARN, Status, Creation Time)
- Extract Model Name from Endpoint Configuration

### 2. IAM Role Identification
- Retrieve the current SageMaker execution role ARN

### 3. SageMaker Version & Role Verification
- Update and verify SageMaker SDK version
- Confirm execution role

### 4. Data Quality Baseline Generation
- Launch SageMaker Processing Job to create `statistics.json` and `constraints.json`

### 5. Monitoring Schedule Health Verification
- Check status of existing monitoring schedules
- List all schedules attached to an endpoint if name is not found

### 6. Model Quality Monitoring Automation
- Create hourly Model Quality Monitor schedule
- Configure Ground Truth input, inference/probability attributes, and dataset format

### 7. Comprehensive Audit & Debugging View
- Detailed breakdown of a monitoring schedule
- S3 path validation (baseline, ground truth, output)
- Full raw configuration printing

---

## Prerequisites

- SageMaker Studio or Jupyter Notebook with `conda_tensorflow2_p310` kernel
- Required packages:
  ```bash
  pip install boto3 sagemaker pandas