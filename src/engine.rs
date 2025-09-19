use crate::types::{ClassificationRequest, ClassificationResponse};
use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Engine {
    async fn classify(&self, request: ClassificationRequest) -> Result<ClassificationResponse>;
}

#[async_trait]
pub trait BatchedEngine: Send + Sync {
    async fn classify_batch(
        &self,
        requests: Vec<ClassificationRequest>,
    ) -> Result<Vec<Result<ClassificationResponse>>>;
}
