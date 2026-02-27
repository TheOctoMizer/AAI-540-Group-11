# SageMaker Model Package Group Snapshot & Governance Audit

## Overview

This notebook provides a **complete, repeatable, and auditable** process to capture and document AWS SageMaker Model Packages and Model Groups.

It automatically:
- Fetches full metadata for a specific Model Package (including Version 2)
- Retrieves Model Package Group details
- Lists all versions with status and approval information
- Generates a comprehensive snapshot (JSON) with inference specifications, model metrics, tags, and comparison tables
- Saves the snapshot locally and uploads it to S3 for long-term governance

**Purpose**: Supports **Model Governance, Compliance, Version Tracking, and Model Card** requirements for the UEBA (User and Entity Behavior Analytics) project.

**Suggested Notebook Title**:  
**AWS SageMaker Model Package Group Snapshot & Governance Audit**

---

## Key Features

- Full snapshot of any Model Package (ARN-based)
- Model Package Group metadata extraction
- Complete version history with approval status
- Detailed inference container information
- Model metrics (quality, bias, explainability)
- Tags and IAM identity tracking
- Side-by-side version comparison table (Pandas)
- Automatic local + S3 storage with timestamped filenames
- Clean, professional audit-ready output

---

## Notebook Structure

### 1. Basic Model Package Snapshot
- Describe single Model Package (Version 2)
- Fetch group details
- List all versions
- Create and upload full JSON snapshot

### 2. Model Group Governance Audit Report
- Simple governance summary
- Version history table with status and creation dates

### 3. Comprehensive Model Package Group Analysis
- Detailed metadata for the group
- Deep dive into every version (inference spec, containers, metrics, tags)
- Version comparison table
- Full JSON snapshot export (local + S3)

---

## Prerequisites

- Python 3.10+
- Required packages:
  ```bash
  pip install boto3 pandas