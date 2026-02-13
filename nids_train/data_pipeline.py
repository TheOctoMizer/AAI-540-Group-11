"""
Comprehensive Data Pipeline for AI NIDS Project

This script processes all CIC-IDS2017 datasets and prepares them for AWS deployment:
1. Data Collection & Cataloging (for Athena tables)
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Splitting (40% train / 10% test / 10% validation / 40% production)
5. Feature Store Preparation (AWS-agnostic Parquet/CSV format)

Usage:
    python data_pipeline.py --data-dir ./nids_dataset --output-dir ./output
    python data_pipeline.py --phase catalog  # Run specific phase only
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DataPipeline:
    """Comprehensive data pipeline for NIDS dataset processing."""
    
    def __init__(self, data_dir: str, output_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.scaler = None
        self.feature_names = None
        
        # Dataset files
        self.dataset_files = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        ]
        
        # Columns to drop (metadata, not features)
        self.drop_cols = [
            "Flow ID", "Source IP", "Source Port", 
            "Destination IP", "Destination Port", 
            "Protocol", "Timestamp"
        ]
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create output directory structure."""
        dirs = [
            self.output_dir / "catalog",
            self.output_dir / "eda",
            self.output_dir / "features" / "train",
            self.output_dir / "features" / "test",
            self.output_dir / "features" / "val",
            self.output_dir / "features" / "production",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load a single dataset file."""
        filepath = self.data_dir / filename
        print(f"Loading {filename}...")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        return df
    
    def phase_1_catalog(self) -> Dict:
        """
        Phase 1: Data Collection & Cataloging
        
        Generates metadata catalog for all datasets (Athena-compatible).
        """
        print("\n" + "="*80)
        print("PHASE 1: DATA COLLECTION & CATALOGING")
        print("="*80)
        
        catalog = {
            "created_at": datetime.now().isoformat(),
            "source": "CIC-IDS2017",
            "datasets": [],
            "feature_names": None,
            "attack_types": set(),
            "total_samples": 0,
            "total_size_mb": 0
        }
        
        for filename in self.dataset_files:
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"WARNING: {filename} not found, skipping...")
                continue
            
            # Load dataset
            df = self.load_dataset(filename)
            
            # Get file size
            size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Get class distribution
            if "Label" in df.columns:
                class_dist = df["Label"].value_counts().to_dict()
                attack_types = set(df["Label"].unique())
                catalog["attack_types"].update(attack_types)
            else:
                class_dist = {}
                attack_types = set()
            
            # Store feature names (first dataset)
            if catalog["feature_names"] is None:
                catalog["feature_names"] = [
                    col for col in df.columns 
                    if col not in self.drop_cols + ["Label"]
                ]
            
            # Dataset metadata
            dataset_meta = {
                "filename": filename,
                "size_mb": round(size_mb, 2),
                "num_samples": len(df),
                "num_features": len(df.columns),
                "class_distribution": class_dist,
                "attack_types": list(attack_types),
                "has_label": "Label" in df.columns
            }
            
            catalog["datasets"].append(dataset_meta)
            catalog["total_samples"] += len(df)
            catalog["total_size_mb"] += size_mb
            
            print(f"  {filename}: {len(df):,} samples, {size_mb:.2f} MB")
        
        # Convert set to list for JSON serialization
        catalog["attack_types"] = sorted(list(catalog["attack_types"]))
        catalog["total_size_mb"] = round(catalog["total_size_mb"], 2)
        catalog["num_datasets"] = len(catalog["datasets"])
        catalog["num_features"] = len(catalog["feature_names"])
        
        # Save catalog
        catalog_path = self.output_dir / "catalog" / "data_catalog.json"
        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)
        
        print(f"\n✓ Catalog saved to {catalog_path}")
        print(f"  Total datasets: {catalog['num_datasets']}")
        print(f"  Total samples: {catalog['total_samples']:,}")
        print(f"  Total size: {catalog['total_size_mb']:.2f} MB")
        print(f"  Attack types: {len(catalog['attack_types'])}")
        
        return catalog
    
    def phase_2_eda(self, generate_plots: bool = True) -> Dict:
        """
        Phase 2: Exploratory Data Analysis
        
        Generates statistical analysis and visualizations.
        """
        print("\n" + "="*80)
        print("PHASE 2: EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Load all datasets
        all_data = []
        for filename in self.dataset_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                df = self.load_dataset(filename)
                all_data.append(df)
        
        # Combine datasets
        print("\nCombining all datasets...")
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total samples: {len(combined_df):,}")
        
        # Class distribution analysis
        print("\nClass Distribution:")
        class_counts = combined_df["Label"].value_counts()
        class_dist = (class_counts / len(combined_df) * 100).to_dict()
        for label, pct in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {class_counts[label]:,} ({pct:.2f}%)")
        
        # Feature statistics
        print("\nCalculating feature statistics...")
        feature_cols = [col for col in combined_df.columns 
                       if col not in self.drop_cols + ["Label"]]
        
        # Remove inf and NaN for statistics
        feature_df = combined_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        feature_stats = {}
        for col in feature_cols:
            clean_data = feature_df[col].dropna()
            if len(clean_data) > 0:
                feature_stats[col] = {
                    "count": int(len(clean_data)),
                    "mean": float(clean_data.mean()),
                    "std": float(clean_data.std()),
                    "min": float(clean_data.min()),
                    "max": float(clean_data.max()),
                    "p25": float(clean_data.quantile(0.25)),
                    "p50": float(clean_data.quantile(0.50)),
                    "p75": float(clean_data.quantile(0.75)),
                    "p95": float(clean_data.quantile(0.95)),
                    "p99": float(clean_data.quantile(0.99)),
                    "missing_count": int(feature_df[col].isna().sum()),
                    "missing_pct": float(feature_df[col].isna().sum() / len(feature_df) * 100)
                }
        
        # Missing values analysis
        missing_analysis = {
            col: {
                "count": int(feature_df[col].isna().sum()),
                "percentage": float(feature_df[col].isna().sum() / len(feature_df) * 100)
            }
            for col in feature_cols
            if feature_df[col].isna().sum() > 0
        }
        
        # Infinite values analysis
        inf_analysis = {}
        for col in feature_cols:
            inf_count = np.isinf(combined_df[col]).sum()
            if inf_count > 0:
                inf_analysis[col] = {
                    "count": int(inf_count),
                    "percentage": float(inf_count / len(combined_df) * 100)
                }
        
        # Create EDA report
        eda_report = {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(combined_df),
            "num_features": len(feature_cols),
            "class_distribution": {k: int(v) for k, v in class_counts.to_dict().items()},
            "class_distribution_pct": class_dist,
            "feature_statistics": feature_stats,
            "missing_values": missing_analysis,
            "infinite_values": inf_analysis
        }
        
        # Save EDA report
        eda_path = self.output_dir / "eda" / "eda_report.json"
        with open(eda_path, "w") as f:
            json.dump(eda_report, f, indent=2)
        
        print(f"\n✓ EDA report saved to {eda_path}")
        
        # Generate visualizations
        if generate_plots:
            print("\nGenerating visualizations...")
            self._generate_eda_plots(combined_df, class_counts, feature_df)
        
        return eda_report
    
    def _generate_eda_plots(self, df: pd.DataFrame, class_counts: pd.Series, 
                           feature_df: pd.DataFrame):
        """Generate EDA visualization plots."""
        
        # 1. Class Distribution Plot
        plt.figure(figsize=(12, 6))
        class_counts_sorted = class_counts.sort_values(ascending=False)
        plt.bar(range(len(class_counts_sorted)), class_counts_sorted.values)
        plt.xticks(range(len(class_counts_sorted)), class_counts_sorted.index, rotation=45, ha='right')
        plt.xlabel('Attack Type')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution Across All Datasets')
        plt.tight_layout()
        plt.savefig(self.output_dir / "eda" / "class_distribution.png", dpi=300)
        plt.close()
        print("  ✓ Class distribution plot saved")
        
        # 2. Top 20 Features Statistics (by variance)
        feature_cols = [col for col in df.columns 
                       if col not in self.drop_cols + ["Label"]]
        feature_variance = feature_df[feature_cols].var().sort_values(ascending=False)
        top_20_features = feature_variance.head(20).index.tolist()
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_20_features):
            clean_data = feature_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_data) > 0:
                axes[idx].hist(clean_data, bins=50, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f"{feature}\n(var={feature_variance[feature]:.2e})", fontsize=8)
                axes[idx].set_xlabel('Value', fontsize=7)
                axes[idx].set_ylabel('Frequency', fontsize=7)
                axes[idx].tick_params(labelsize=6)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "eda" / "feature_statistics.png", dpi=300)
        plt.close()
        print("  ✓ Feature statistics plot saved")
        
        # 3. Correlation matrix (top 15 features)
        top_15_features = feature_variance.head(15).index.tolist()
        corr_df = feature_df[top_15_features].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(corr_df) > 0:
            plt.figure(figsize=(14, 12))
            correlation_matrix = corr_df.corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Matrix (Top 15 Features by Variance)')
            plt.tight_layout()
            plt.savefig(self.output_dir / "eda" / "feature_correlation.png", dpi=300)
            plt.close()
            print("  ✓ Feature correlation plot saved")
    
    def phase_3_feature_engineering(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase 3: Feature Engineering
        
        Applies consistent preprocessing pipeline to all datasets.
        Returns combined features and labels.
        """
        print("\n" + "="*80)
        print("PHASE 3: FEATURE ENGINEERING")
        print("="*80)
        
        all_features = []
        all_labels = []
        
        for filename in self.dataset_files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                continue
            
            # Load dataset
            df = self.load_dataset(filename)
            
            # Extract labels
            if "Label" in df.columns:
                labels = df["Label"].values
            else:
                labels = np.array(["UNKNOWN"] * len(df))
            
            # Drop metadata columns
            feature_df = df.drop(columns=[c for c in self.drop_cols + ["Label"] 
                                         if c in df.columns], errors="ignore")
            
            # Clean data (remove inf and NaN)
            original_len = len(feature_df)
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Filter labels to match cleaned data
            if len(feature_df) < original_len:
                kept_indices = feature_df.index
                labels = labels[kept_indices]
            
            # Reset index
            feature_df = feature_df.reset_index(drop=True)
            
            # Store feature names (first dataset)
            if self.feature_names is None:
                self.feature_names = feature_df.columns.tolist()
            
            # Reorder columns to ensure consistency
            feature_df = feature_df[self.feature_names]
            
            all_features.append(feature_df.values)
            all_labels.append(labels)
            
            print(f"  {filename}: {len(feature_df):,} samples processed")
        
        # Combine all features and labels
        X = np.vstack(all_features).astype(np.float32)
        y = np.concatenate(all_labels)
        
        print(f"\nTotal processed samples: {len(X):,}")
        print(f"Total features: {X.shape[1]}")
        
        # Fit scaler on all data
        print("\nFitting MinMaxScaler...")
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Save scaler parameters
        scaler_params = {
            "scale": self.scaler.scale_.tolist(),
            "min": self.scaler.min_.tolist(),
            "n_features": self.scaler.n_features_in_,
            "feature_columns": self.feature_names
        }
        
        scaler_path = self.output_dir / "scaler_params.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"✓ Scaler parameters saved to {scaler_path}")
        
        return X_scaled, y
    
    def phase_4_split_data(self, X: np.ndarray, y: np.ndarray, 
                           train_ratio: float = 0.4,
                           test_ratio: float = 0.1,
                           val_ratio: float = 0.1,
                           prod_ratio: float = 0.4) -> Dict:
        """
        Phase 4: Data Splitting
        
        Splits data into train/test/val/production sets with stratification.
        """
        print("\n" + "="*80)
        print("PHASE 4: DATA SPLITTING")
        print("="*80)
        
        # Validate ratios
        total_ratio = train_ratio + test_ratio + val_ratio + prod_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        print(f"Split ratios: {train_ratio:.0%} train / {test_ratio:.0%} test / "
              f"{val_ratio:.0%} val / {prod_ratio:.0%} production")
        
        # First split: separate production data
        X_temp, X_prod, y_temp, y_prod = train_test_split(
            X, y, test_size=prod_ratio, random_state=self.seed, stratify=y
        )
        
        # Calculate remaining ratios
        remaining = 1.0 - prod_ratio
        train_ratio_adj = train_ratio / remaining
        test_val_ratio = (test_ratio + val_ratio) / remaining
        
        # Second split: separate training data
        X_train, X_test_val, y_train, y_test_val = train_test_split(
            X_temp, y_temp, test_size=test_val_ratio, 
            random_state=self.seed, stratify=y_temp
        )
        
        # Third split: separate test and validation
        test_ratio_adj = test_ratio / (test_ratio + val_ratio)
        X_test, X_val, y_test, y_val = train_test_split(
            X_test_val, y_test_val, test_size=(1-test_ratio_adj),
            random_state=self.seed, stratify=y_test_val
        )
        
        splits = {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
            "val": (X_val, y_val),
            "production": (X_prod, y_prod)
        }
        
        print("\nSplit sizes:")
        for split_name, (X_split, y_split) in splits.items():
            print(f"  {split_name}: {len(X_split):,} samples ({len(X_split)/len(X)*100:.2f}%)")
            
            # Show class distribution
            unique, counts = np.unique(y_split, return_counts=True)
            print(f"    Classes: {len(unique)}")
            for label, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      {label}: {count:,}")
        
        return splits
    
    def phase_5_feature_store(self, splits: Dict):
        """
        Phase 5: Feature Store Preparation
        
        Saves processed features in AWS-agnostic format (Parquet + CSV).
        """
        print("\n" + "="*80)
        print("PHASE 5: FEATURE STORE PREPARATION")
        print("="*80)
        
        for split_name, (X_split, y_split) in splits.items():
            print(f"\nSaving {split_name} split...")
            
            split_dir = self.output_dir / "features" / split_name
            
            # Create DataFrame
            df_features = pd.DataFrame(X_split, columns=self.feature_names)
            df_labels = pd.DataFrame({"label": y_split})
            
            # Save as Parquet (efficient columnar format)
            parquet_path = split_dir / "features.parquet"
            df_features.to_parquet(parquet_path, index=False, compression='snappy')
            print(f"  ✓ Parquet: {parquet_path}")
            
            # Save as CSV (human-readable, Athena-compatible)
            csv_path = split_dir / "features.csv"
            df_features.to_csv(csv_path, index=False)
            print(f"  ✓ CSV: {csv_path}")
            
            # Save labels separately
            labels_path = split_dir / "labels.csv"
            df_labels.to_csv(labels_path, index=False)
            print(f"  ✓ Labels: {labels_path}")
            
            # Create metadata
            unique_labels, counts = np.unique(y_split, return_counts=True)
            class_distribution = {
                str(label): int(count) 
                for label, count in zip(unique_labels, counts)
            }
            
            metadata = {
                "split": split_name,
                "num_samples": int(len(X_split)),
                "num_features": int(X_split.shape[1]),
                "feature_names": self.feature_names,
                "class_distribution": class_distribution,
                "preprocessing": {
                    "scaler": "MinMaxScaler",
                    "scaler_params_file": "../scaler_params.json",
                    "dropped_columns": self.drop_cols
                },
                "created_at": datetime.now().isoformat(),
                "file_formats": ["parquet", "csv"]
            }
            
            metadata_path = split_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Metadata: {metadata_path}")
        
        print("\n✓ Feature store preparation complete!")
    
    def run_all_phases(self, generate_plots: bool = True):
        """Run all pipeline phases sequentially."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("RUNNING COMPLETE DATA PIPELINE")
        print("="*80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Random seed: {self.seed}")
        
        # Phase 1: Catalog
        catalog = self.phase_1_catalog()
        
        # Phase 2: EDA
        eda_report = self.phase_2_eda(generate_plots=generate_plots)
        
        # Phase 3: Feature Engineering
        X, y = self.phase_3_feature_engineering()
        
        # Phase 4: Data Splitting
        splits = self.phase_4_split_data(X, y)
        
        # Phase 5: Feature Store
        self.phase_5_feature_store(splits)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"\nOutputs saved to: {self.output_dir}")
        print("\nNext steps:")
        print("  1. Upload catalog/data_catalog.json to S3 for Athena table creation")
        print("  2. Review eda/eda_report.json and visualizations")
        print("  3. Upload features/* to S3 Feature Store")
        print("  4. Use scaler_params.json for inference preprocessing")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Data Pipeline for AI NIDS Project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./nids_dataset",
        help="Directory containing raw CSV datasets"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save pipeline outputs"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "catalog", "eda", "features", "split", "store"],
        default="all",
        help="Pipeline phase to run (default: all)"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.4,
        help="Training data ratio (default: 0.4)"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test data ratio (default: 0.1)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation data ratio (default: 0.1)"
    )
    
    parser.add_argument(
        "--prod-ratio",
        type=float,
        default=0.4,
        help="Production data ratio (default: 0.4)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Run specified phase(s)
    if args.phase == "all":
        pipeline.run_all_phases(generate_plots=not args.no_plots)
    elif args.phase == "catalog":
        pipeline.phase_1_catalog()
    elif args.phase == "eda":
        pipeline.phase_2_eda(generate_plots=not args.no_plots)
    elif args.phase == "features":
        X, y = pipeline.phase_3_feature_engineering()
        print(f"\nFeatures shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
    elif args.phase == "split":
        X, y = pipeline.phase_3_feature_engineering()
        splits = pipeline.phase_4_split_data(
            X, y,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            prod_ratio=args.prod_ratio
        )
    elif args.phase == "store":
        X, y = pipeline.phase_3_feature_engineering()
        splits = pipeline.phase_4_split_data(X, y)
        pipeline.phase_5_feature_store(splits)


if __name__ == "__main__":
    main()
