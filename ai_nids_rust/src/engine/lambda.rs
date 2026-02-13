use anyhow::{Context, Result};
use aws_sdk_lambda::Client as LambdaClient;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct AttackInfo {
    pub attack_type: String,
    pub confidence: f32,
    pub mse_error: f32,
    pub timestamp: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LambdaResponse {
    #[serde(rename = "statusCode")]
    pub status_code: i32,
    pub body: String,
}

pub struct LambdaTrigger {
    client: LambdaClient,
    function_name: String,
}

impl LambdaTrigger {
    pub async fn new(function_name: String) -> Result<Self> {
        log::info!("Initializing Lambda trigger...");
        log::info!("  Function name: {}", function_name);

        let config = aws_config::load_from_env().await;
        let client = LambdaClient::new(&config);

        log::info!("Lambda client initialized successfully");

        Ok(Self {
            client,
            function_name,
        })
    }

    pub async fn trigger_shutdown(&self, attack_info: AttackInfo) -> Result<()> {
        let start_time = Instant::now();

        log::warn!("TRIGGERING VM SHUTDOWN!");
        log::warn!("  Attack type: {}", attack_info.attack_type);
        log::warn!("  Confidence: {:.4}", attack_info.confidence);
        log::warn!("  MSE error: {:.6}", attack_info.mse_error);

        let payload =
            serde_json::to_vec(&attack_info).context("Failed to serialize attack info")?;

        let response = self
            .client
            .invoke()
            .function_name(&self.function_name)
            .payload(payload.into())
            .send()
            .await
            .context("Failed to invoke Lambda function")?;

        let invoke_time = start_time.elapsed();

        // Parse response
        if let Some(payload) = response.payload() {
            let result: LambdaResponse = serde_json::from_slice(payload.as_ref())
                .context("Failed to parse Lambda response")?;

            log::info!("Lambda invoked successfully in {:?}", invoke_time);
            log::info!("  Status code: {}", result.status_code);
            log::info!("  Response: {}", result.body);
        } else {
            log::warn!("Lambda invoked but no response payload");
        }

        Ok(())
    }
}
