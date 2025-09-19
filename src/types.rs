use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct ClassificationRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ClassificationResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub data: Vec<ClassificationData>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ClassificationData {
    pub index: usize,
    pub label: String,
    pub probs: Vec<f64>,
    pub num_classes: usize,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    pub completion_tokens: u32,
    pub prompt_tokens_details: Option<serde_json::Value>,
}
