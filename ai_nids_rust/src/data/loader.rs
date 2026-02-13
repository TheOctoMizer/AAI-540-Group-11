use anyhow::{Context, Result};
use polars::prelude::*;

#[derive(Debug, Clone)]
pub struct ProductionDataset {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<String>,
}

impl ProductionDataset {
    pub fn load_csv(features_path: &str, labels_path: &str) -> Result<Self> {
        log::info!("Loading production dataset from CSV...");
        log::info!("  Features: {}", features_path);
        log::info!("  Labels: {}", labels_path);

        // Load features
        let features_df = CsvReader::from_path(features_path)
            .context("Failed to open features CSV")?
            .finish()
            .context("Failed to parse features CSV")?;

        // Load labels
        let labels_df = CsvReader::from_path(labels_path)
            .context("Failed to open labels CSV")?
            .finish()
            .context("Failed to parse labels CSV")?;

        // Convert features to Vec<Vec<f32>>
        let num_rows = features_df.height();
        let num_cols = features_df.width();

        log::info!(
            "Dataset loaded: {} samples, {} features",
            num_rows,
            num_cols
        );

        let mut features = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let mut row_features = Vec::with_capacity(num_cols);
            for col_idx in 0..num_cols {
                let col = &features_df.get_columns()[col_idx];
                let value = col
                    .get(row_idx)
                    .context("Failed to get feature value")?
                    .try_extract::<f64>()
                    .unwrap_or(0.0) as f32;
                row_features.push(value);
            }
            features.push(row_features);
        }

        // Extract labels
        let label_col = labels_df
            .column("label")
            .context("No 'label' column in labels CSV")?;

        let labels: Vec<String> = label_col
            .str()
            .context("Label column is not string type")?
            .into_iter()
            .map(|opt: Option<&str>| opt.unwrap_or("UNKNOWN").to_string())
            .collect();

        log::info!("Dataset parsing complete");

        Ok(Self { features, labels })
    }

    pub fn load_parquet(features_path: &str, labels_path: &str) -> Result<Self> {
        log::info!("Loading production dataset from Parquet...");
        log::info!("  Features: {}", features_path);
        log::info!("  Labels: {}", labels_path);

        // Load features
        let features_file =
            std::fs::File::open(features_path).context("Failed to open features Parquet")?;
        let features_df = ParquetReader::new(features_file)
            .finish()
            .context("Failed to parse features Parquet")?;

        // Load labels
        let labels_file = std::fs::File::open(labels_path).context("Failed to open labels CSV")?;
        let labels_df = CsvReader::new(labels_file)
            .finish()
            .context("Failed to parse labels CSV")?;

        // Convert features to Vec<Vec<f32>>
        let num_rows = features_df.height();
        let num_cols = features_df.width();

        log::info!(
            "Dataset loaded: {} samples, {} features",
            num_rows,
            num_cols
        );

        let mut features = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let mut row_features = Vec::with_capacity(num_cols);
            for col_idx in 0..num_cols {
                let col = &features_df.get_columns()[col_idx];
                let value = col
                    .get(row_idx)
                    .context("Failed to get feature value")?
                    .try_extract::<f64>()
                    .unwrap_or(0.0) as f32;
                row_features.push(value);
            }
            features.push(row_features);
        }

        // Extract labels
        let label_col = labels_df
            .column("label")
            .context("No 'label' column in labels CSV")?;

        let labels: Vec<String> = label_col
            .str()
            .context("Label column is not string type")?
            .into_iter()
            .map(|opt: Option<&str>| opt.unwrap_or("UNKNOWN").to_string())
            .collect();

        log::info!("Dataset parsing complete");

        Ok(Self { features, labels })
    }

    pub fn filter_by_label(&self, target_label: &str) -> Self {
        let mut filtered_features = Vec::new();
        let mut filtered_labels = Vec::new();

        for (features, label) in self.features.iter().zip(self.labels.iter()) {
            if label == target_label {
                filtered_features.push(features.clone());
                filtered_labels.push(label.clone());
            }
        }

        log::info!(
            "Filtered dataset: {} samples with label '{}'",
            filtered_features.len(),
            target_label
        );

        Self {
            features: filtered_features,
            labels: filtered_labels,
        }
    }

    pub fn sample(&self, count: usize) -> Self {
        let actual_count = count.min(self.features.len());

        Self {
            features: self.features[..actual_count].to_vec(),
            labels: self.labels[..actual_count].to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}
