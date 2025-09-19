mod batched_engine;
mod config;
mod deberta_engine;
mod engine;
mod types;

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
};
use axum_prometheus::PrometheusMetricLayer;
use clap::Parser;
use metrics::counter;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

use batched_engine::BatchedEngineWrapper;
use config::{BatchConfig, Config};
use deberta_engine::{DebertaBatchedEngine, DebertaConfig};
use engine::Engine;
use types::{ClassificationRequest, ClassificationResponse, Usage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,inference_server=debug".into()),
        )
        .init();

    let config = Config::parse();
    tracing::info!("Starting inference server with config: {:?}", config);

    // Validate that either model_id or model_path is provided
    if config.model_id.is_none() && config.model_path.is_none() {
        anyhow::bail!("Either --model-id or --model-path must be provided");
    }

    let batch_config = BatchConfig::from(&config);

    let deberta_config = DebertaConfig {
        model_id: config.model_id.clone(),
        model_path: config.model_path.clone(),
        revision: config.model_revision.clone(),
        use_pth: config.use_pth,
        cpu: config.cpu_only,
        max_sequence_length: config.max_sequence_length,
        id2label: config.parse_id2label(),
    };

    tracing::info!("Loading DeBERTa model...");
    let deberta_engine = DebertaBatchedEngine::new(deberta_config).await?;
    tracing::info!("Model loaded successfully");

    let (engine, processor) = BatchedEngineWrapper::new(batch_config.clone(), deberta_engine);
    tracing::info!("Batch engine wrapper created");

    // Spawn background task to process batches
    tokio::spawn(async move {
        tracing::info!("Starting batch processor");
        if let Err(e) = processor.run_forever().await {
            tracing::error!("Batch processor error: {}", e);
        }
    });

    let (prometheus_layer, metric_handle) = PrometheusMetricLayer::pair();

    let app = Router::new()
        .route("/classify", post(classify_handler))
        .route("/metrics", get(|| async move { metric_handle.render() }))
        .layer(prometheus_layer)
        .layer(TraceLayer::new_for_http())
        .with_state(AppState::new(Arc::new(engine)));

    let listener = TcpListener::bind(&config.server_address()).await?;
    tracing::info!("Server running on http://{}", config.server_address());
    tracing::info!(
        "Batch size: {}, Tick duration: {:?}",
        batch_config.batch_size,
        batch_config.tick_duration
    );

    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    engine: Arc<dyn Engine + Send + Sync>,
}

impl AppState {
    fn new(engine: Arc<dyn Engine + Send + Sync>) -> Self {
        Self { engine }
    }
}

#[tracing::instrument(skip(state, request), fields(input_count = request.input.len(), model = %request.model))]
async fn classify_handler(
    State(state): State<AppState>,
    Json(request): Json<ClassificationRequest>,
) -> Result<Json<ClassificationResponse>, StatusCode> {
    counter!("classification_requests_total").increment(1);
    tracing::info!("Processing classification request");

    // Split the request into individual single-string requests
    let individual_requests: Vec<ClassificationRequest> = request
        .input
        .iter()
        .map(|text| ClassificationRequest {
            model: request.model.clone(),
            input: vec![text.clone()],
        })
        .collect();

    // Process all individual requests concurrently
    let futures = individual_requests
        .into_iter()
        .map(|req| state.engine.classify(req));

    let results = futures::future::join_all(futures).await;

    // Check for any errors and collect successful responses
    let mut all_data = Vec::new();
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;

    for (index, result) in results.into_iter().enumerate() {
        match result {
            Ok(response) => {
                // Add the classification data, adjusting the index to match original position
                for mut data in response.data {
                    data.index = index;
                    all_data.push(data);
                }
                total_prompt_tokens += response.usage.prompt_tokens;
                total_completion_tokens += response.usage.completion_tokens;
            }
            Err(e) => {
                tracing::error!(input_index = index, error = %e, "Classification failed");
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    }

    // Create the merged response
    let merged_response = ClassificationResponse {
        id: format!("classify-{}", uuid::Uuid::new_v4().simple()),
        object: "list".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: request.model,
        data: all_data,
        usage: Usage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens: total_completion_tokens,
            prompt_tokens_details: None,
        },
    };

    tracing::info!("Classification completed successfully");
    Ok(Json(merged_response))
}
