use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    /// Batch size for processing requests
    #[arg(long, env = "BATCH_SIZE", default_value = "8")]
    pub batch_size: usize,

    /// Tick duration in milliseconds for batch processing
    #[arg(long, env = "TICK_DURATION_MS", default_value = "100")]
    pub tick_duration_ms: u64,

    /// Server host to bind to
    #[arg(long, env = "HOST", default_value = "127.0.0.1")]
    pub host: String,

    /// Server port to bind to
    #[arg(long, env = "PORT", default_value = "8000")]
    pub port: u16,

    /// Model ID from Hugging Face Hub
    #[arg(long, env = "MODEL_ID")]
    pub model_id: Option<String>,

    /// Local path to model directory
    #[arg(long, env = "MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Model revision/branch on Hugging Face
    #[arg(long, env = "MODEL_REVISION", default_value = "main")]
    pub model_revision: String,

    /// Use PyTorch weights instead of safetensors
    #[arg(long, env = "USE_PTH")]
    pub use_pth: bool,

    /// Run on CPU instead of GPU
    #[arg(long, env = "CPU_ONLY")]
    pub cpu_only: bool,

    /// Maximum sequence length allowed
    #[arg(long, env = "MAX_SEQUENCE_LENGTH", default_value = "512")]
    pub max_sequence_length: usize,

    /// Labels mapping in format "0=No Claim,1=Claim"
    #[arg(long, env = "ID2LABEL")]
    pub id2label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub tick_duration: Duration,
}

impl From<&Config> for BatchConfig {
    fn from(config: &Config) -> Self {
        Self {
            batch_size: config.batch_size,
            tick_duration: Duration::from_millis(config.tick_duration_ms),
        }
    }
}

impl Config {
    pub fn parse_id2label(&self) -> Option<HashMap<u32, String>> {
        self.id2label.as_ref().map(|labels| {
            labels
                .split(',')
                .filter_map(|pair| {
                    let mut parts = pair.split('=');
                    let id = parts.next()?.parse().ok()?;
                    let label = parts.next()?.to_string();
                    Some((id, label))
                })
                .collect()
        })
    }

    pub fn server_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}
