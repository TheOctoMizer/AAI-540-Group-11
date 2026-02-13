use anyhow::{Context, Result};
use aws_sdk_sagemakerruntime::Client as SageMakerClient;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::classifier::ClassificationResult;

#[derive(Debug, Clone, Serialize)]
pub struct AutoencoderRequest {
    pub features: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AutoencoderResponse {
    pub reconstruction: Vec<Vec<f32>>,
    pub encoded: Vec<Vec<f32>>,
    pub mse_error: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct XGBoostRequest {
    pub encoded: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct XGBoostResponse {
    pub class_ids: Vec<i32>,
    pub labels: Vec<String>,
    pub confidences: Vec<f32>,
    pub probabilities: Option<Vec<Vec<f32>>>,
}

pub struct SageMakerRuntime {
    client: SageMakerClient,
    autoencoder_endpoint: String,
    xgboost_endpoint: String,
}

impl SageMakerRuntime {
    pub async fn new(autoencoder_endpoint: String, xgboost_endpoint: String) -> Result<Self> {
        log::info!("Initializing SageMaker runtime...");
        log::info!("  Autoencoder endpoint: {}", autoencoder_endpoint);
        log::info!("  XGBoost endpoint: {}", xgboost_endpoint);

        let config = aws_config::load_from_env().await;
        let client = SageMakerClient::new(&config);

        log::info!("SageMaker client initialized successfully");

        Ok(Self {
            client,
            autoencoder_endpoint,
            xgboost_endpoint,
        })
    }

    pub async fn invoke_autoencoder(&self, features: Vec<f32>) -> Result<(Vec<f32>, f32)> {
        let start_time = Instant::now();

        log::debug!(
            "Invoking autoencoder endpoint with {} features",
            features.len()
        );

        let request = AutoencoderRequest { features };
        let payload =
            serde_json::to_vec(&request).context("Failed to serialize autoencoder request")?;

        let response = self
            .client
            .invoke_endpoint()
            .endpoint_name(&self.autoencoder_endpoint)
            .content_type("application/json")
            .body(payload.into())
            .send()
            .await
            .context("Failed to invoke autoencoder endpoint")?;

        let binding = response.body.clone();
        let bytes = binding.as_ref().context("Empty body")?.as_ref();
        let result: AutoencoderResponse =
            serde_json::from_slice(bytes).context("Failed to parse autoencoder response")?;

        let inference_time = start_time.elapsed();
        log::debug!("Autoencoder inference completed in {:?}", inference_time);

        // Extract first sample results
        let encoded = result
            .encoded
            .get(0)
            .context("No encoded features in response")?
            .clone();
        let mse_error = *result
            .mse_error
            .get(0)
            .context("No MSE error in response")?;

        log::debug!(
            "MSE error: {:.6}, Encoded dim: {}",
            mse_error,
            encoded.len()
        );

        Ok((encoded, mse_error))
    }

    pub async fn invoke_xgboost(&self, encoded: Vec<f32>) -> Result<ClassificationResult> {
        let start_time = Instant::now();

        log::debug!(
            "Invoking XGBoost endpoint with {} encoded features",
            encoded.len()
        );

        let request = XGBoostRequest { encoded };
        let payload =
            serde_json::to_vec(&request).context("Failed to serialize XGBoost request")?;

        let response = self
            .client
            .invoke_endpoint()
            .endpoint_name(&self.xgboost_endpoint)
            .content_type("application/json")
            .body(payload.into())
            .send()
            .await
            .context("Failed to invoke XGBoost endpoint")?;

        let binding = response.body.clone();
        let bytes = binding.as_ref().context("Empty body")?.as_ref();
        let result: XGBoostResponse =
            serde_json::from_slice(bytes).context("Failed to parse XGBoost response")?;

        let inference_time = start_time.elapsed();
        log::debug!("XGBoost inference completed in {:?}", inference_time);

        // Extract first sample results
        let label = result
            .labels
            .get(0)
            .context("No label in response")?
            .clone();
        let confidence = *result
            .confidences
            .get(0)
            .context("No confidence in response")?;
        let class_id = *result.class_ids.get(0).context("No class ID in response")? as usize;

        log::debug!("Predicted: {} (confidence: {:.4})", label, confidence);

        Ok(ClassificationResult {
            label,
            confidence,
            class_id,
        })
    }
}
