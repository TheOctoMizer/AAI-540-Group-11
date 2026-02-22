mod data;
mod engine;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tower_http::trace::TraceLayer;

use data::loader::ProductionDataset;
use engine::encoder::Autoencoder;
use engine::lambda::{AttackInfo, LambdaTrigger};
use engine::sagemaker::SageMakerRuntime;

#[derive(Parser)]
#[command(name = "ai_nids_rust")]
#[command(about = "AI-based Network Intrusion Detection System with SageMaker")]
struct Args {
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[arg(long, default_value = "models/autoencoder_fp32.onnx")]
    autoencoder_model: String,

    #[arg(long, default_value = "nids-autoencoder")]
    autoencoder_endpoint: String,

    #[arg(long, default_value = "nids-xgboost")]
    xgboost_endpoint: String,

    #[arg(long, default_value = "nids-vm-shutdown")]
    lambda_function: String,

    #[arg(long, default_value = "../nids_train/output/features/production")]
    production_data_dir: PathBuf,

    #[arg(long, default_value = "http://localhost:8080")]
    go_server_url: String,

    #[arg(long, default_value = "0.0567")]
    threshold: f32,
}

#[derive(Clone)]
struct AppState {
    local_ae: Arc<Autoencoder>,
    sagemaker: Arc<SageMakerRuntime>,
    lambda: Arc<LambdaTrigger>,
    production_data_dir: PathBuf,
    go_server_url: String,
    threshold: f32,
}

#[derive(Debug, Deserialize)]
struct TriggerRequest {
    attack_type: Option<String>,
    count: Option<usize>,
}

#[derive(Debug, Serialize)]
struct TriggerResponse {
    total_samples: usize,
    benign_count: usize,
    anomalous_count: usize,
    malicious_count: usize,
    lambda_triggered: bool,
    detections: Vec<DetectionSummary>,
}

#[derive(Debug, Serialize)]
struct DetectionSummary {
    sample_index: usize,
    true_label: String,
    mse_error: f32,
    is_anomalous: bool,
    predicted_label: Option<String>,
    confidence: Option<f32>,
    action: String,
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "NIDS",
        "version": "1.0.0"
    }))
}

async fn ping_middleware(State(state): State<AppState>) -> Result<Response, AppError> {
    log::info!("Ping middleware: forwarding to Go server");

    let client = reqwest::Client::new();
    let response = client
        .get(format!("{}/ping", state.go_server_url))
        .send()
        .await
        .context("Failed to forward ping to Go server")?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    log::info!("Go server responded: {} - {}", status, body);

    Ok((StatusCode::OK, body).into_response())
}

async fn trigger_simulation(
    State(state): State<AppState>,
    Json(request): Json<TriggerRequest>,
) -> Result<Json<TriggerResponse>, AppError> {
    let start_time = Instant::now();

    log::info!("{}", "=".repeat(80));
    log::info!("TRIGGER SIMULATION STARTED");
    log::info!("{}", "=".repeat(80));

    // Load production dataset
    let features_path = state
        .production_data_dir
        .join("features.parquet")
        .to_str()
        .unwrap()
        .to_string();
    let labels_path = state
        .production_data_dir
        .join("labels.csv")
        .to_str()
        .unwrap()
        .to_string();

    log::info!("Loading production dataset...");
    let mut dataset = ProductionDataset::load_parquet(&features_path, &labels_path)
        .context("Failed to load production dataset")?;

    // Filter by attack type if specified
    if let Some(ref attack_type) = request.attack_type {
        log::info!("Filtering for attack type: {}", attack_type);
        dataset = dataset.filter_by_label(attack_type);
    }

    // Limit sample count
    let count = request.count.unwrap_or(100).min(1000);
    dataset = dataset.sample(count);

    log::info!("Processing {} samples...", dataset.len());

    let mut benign_count = 0;
    let mut anomalous_count = 0;
    let mut malicious_count = 0;
    let mut lambda_triggered = false;
    let mut detections = Vec::new();

    // Process each sample
    for (idx, (features, true_label)) in dataset
        .features
        .iter()
        .zip(dataset.labels.iter())
        .enumerate()
    {
        log::debug!("Processing sample {}/{}", idx + 1, dataset.len());

        // Step 1: Call local autoencoder
        let (encoded, mse_error) = state
            .local_ae
            .encode(features.clone())
            .context("Failed to run local autoencoder inference")?;

        // Step 2: Check threshold
        let is_anomalous = mse_error >= state.threshold;

        let (predicted_label, confidence, action) = if is_anomalous {
            anomalous_count += 1;

            // Step 3: Classify attack type via XGBoost (SageMaker)
            // XGBoost only sees encoded features from the autoencoder.
            // All samples at this point are anomalous — XGBoost identifies
            // the specific attack class (no BENIGN class in the model).
            let classification = state
                .sagemaker
                .invoke_xgboost(encoded)
                .await
                .context("Failed to invoke XGBoost")?;

            malicious_count += 1;

            // Step 4: Trigger Lambda on first malicious detection
            if !lambda_triggered {
                log::warn!(
                    "ATTACK DETECTED: {} (confidence: {:.2}%, MSE: {:.6}) — TRIGGERING LAMBDA",
                    classification.label,
                    classification.confidence * 100.0,
                    mse_error
                );

                let attack_info = AttackInfo {
                    attack_type: classification.label.clone(),
                    confidence: classification.confidence,
                    mse_error,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };

                state
                    .lambda
                    .trigger_shutdown(attack_info)
                    .await
                    .context("Failed to trigger Lambda")?;

                lambda_triggered = true;
            } else {
                log::warn!(
                    "ATTACK DETECTED: {} (confidence: {:.2}%) — Lambda already triggered",
                    classification.label,
                    classification.confidence * 100.0,
                );
            }

            (
                Some(classification.label.clone()),
                Some(classification.confidence),
                format!("ATTACK: {}", classification.label),
            )
        } else {
            benign_count += 1;
            (None, None, "BENIGN - Passed".to_string())
        };

        detections.push(DetectionSummary {
            sample_index: idx,
            true_label: true_label.clone(),
            mse_error,
            is_anomalous,
            predicted_label,
            confidence,
            action,
        });

        if idx % 10 == 0 {
            log::info!(
                "Progress: {}/{} - Benign: {}, Anomalous: {}, Malicious: {}",
                idx + 1,
                dataset.len(),
                benign_count,
                anomalous_count,
                malicious_count
            );
        }
    }

    let total_time = start_time.elapsed();

    log::info!("{}", "=".repeat(80));
    log::info!("SIMULATION COMPLETE");
    log::info!("{}", "=".repeat(80));
    log::info!("Total samples: {}", dataset.len());
    log::info!("Benign: {}", benign_count);
    log::info!("Anomalous: {}", anomalous_count);
    log::info!("Malicious: {}", malicious_count);
    log::info!("Lambda triggered: {}", lambda_triggered);
    log::info!("Total time: {:?}", total_time);
    log::info!("{}", "=".repeat(80));

    Ok(Json(TriggerResponse {
        total_samples: dataset.len(),
        benign_count,
        anomalous_count,
        malicious_count,
        lambda_triggered,
        detections,
    }))
}

// Error handling
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        log::error!("Request error: {:#}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Internal server error: {}", self.0),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

fn init_logger(level: &str) -> Result<()> {
    let level = match level.to_lowercase().as_str() {
        "error" => log::LevelFilter::Error,
        "warn" => log::LevelFilter::Warn,
        "info" => log::LevelFilter::Info,
        "debug" => log::LevelFilter::Debug,
        "trace" => log::LevelFilter::Trace,
        _ => log::LevelFilter::Info,
    };

    env_logger::Builder::from_default_env()
        .filter_level(level)
        .init();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    init_logger(&args.log_level)?;

    log::info!("{}", "=".repeat(80));
    log::info!("AI NIDS - Network Intrusion Detection System");
    log::info!("{}", "=".repeat(80));
    log::info!("Autoencoder endpoint: {}", args.autoencoder_endpoint);
    log::info!("XGBoost endpoint: {}", args.xgboost_endpoint);
    log::info!("Lambda function: {}", args.lambda_function);
    log::info!("Production data: {:?}", args.production_data_dir);
    log::info!("Go server URL: {}", args.go_server_url);
    log::info!("Detection threshold: {:.6}", args.threshold);
    log::info!("{}", "=".repeat(80));

    // Initialize local autoencoder
    log::info!("Initializing local autoencoder...");
    let local_ae = Autoencoder::new(&args.autoencoder_model)?;

    // Initialize SageMaker runtime
    log::info!("Initializing SageMaker runtime (XGBoost only)...");
    let sagemaker = SageMakerRuntime::new(
        args.autoencoder_endpoint.clone(),
        args.xgboost_endpoint.clone(),
    )
    .await?;

    // Initialize Lambda trigger
    log::info!("Initializing Lambda trigger...");
    let lambda = LambdaTrigger::new(args.lambda_function.clone()).await?;

    // Create app state
    let state = AppState {
        local_ae: Arc::new(local_ae),
        sagemaker: Arc::new(sagemaker),
        lambda: Arc::new(lambda),
        production_data_dir: args.production_data_dir,
        go_server_url: args.go_server_url,
        threshold: args.threshold,
    };

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/ping", get(ping_middleware))
        .route("/trigger", post(trigger_simulation))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let addr = "0.0.0.0:3000";
    log::info!("Starting server on {}", addr);
    log::info!("Endpoints:");
    log::info!("  GET  /health  - Health check");
    log::info!("  GET  /ping    - Ping middleware (forwards to Go server)");
    log::info!("  POST /trigger - Trigger attack simulation");
    log::info!("{}", "=".repeat(80));

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
