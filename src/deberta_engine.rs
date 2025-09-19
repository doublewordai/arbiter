use anyhow::{Result, bail};
use async_trait::async_trait;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::softmax;
use candle_transformers::models::debertav2::{
    Config as DebertaV2Config, DebertaV2SeqClassificationModel, Id2Label,
};
use chrono::Utc;
use hf_hub::{Repo, RepoType, api::tokio::Api};
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};
use uuid::Uuid;

use crate::engine::BatchedEngine;
use crate::types::{ClassificationData, ClassificationRequest, ClassificationResponse, Usage};

pub struct DebertaBatchedEngine {
    model: DebertaV2SeqClassificationModel,
    tokenizer: Tokenizer,
    device: Device,
    id2label: Id2Label,
}

#[derive(Debug, Clone)]
pub struct DebertaConfig {
    pub model_id: Option<String>,
    pub model_path: Option<PathBuf>,
    pub revision: String,
    pub use_pth: bool,
    pub cpu: bool,
    pub max_sequence_length: usize,
    pub id2label: Option<HashMap<u32, String>>,
}

impl Default for DebertaConfig {
    fn default() -> Self {
        Self {
            model_id: None,
            model_path: None,
            revision: "main".to_string(),
            use_pth: false,
            cpu: false,
            max_sequence_length: 512,
            id2label: None,
        }
    }
}

impl DebertaBatchedEngine {
    fn device(cpu: bool) -> Result<Device> {
        if cpu {
            Ok(Device::Cpu)
        } else if metal_is_available() {
            tracing::info!("Using metal acceleration");
            Ok(Device::new_metal(0)?)
        } else if cuda_is_available() {
            tracing::info!("Using CUDA GPU acceleration");
            Ok(Device::new_cuda(0)?)
        } else {
            tracing::info!(
                "CUDA not available, running on CPU. To run on GPU, build with `--features cuda`"
            );
            Ok(Device::Cpu)
        }
    }

    #[tracing::instrument(skip(config), fields(model_id = ?config.model_id, cpu = config.cpu))]
    pub async fn new(config: DebertaConfig) -> Result<Self> {
        let device = Self::device(config.cpu)?;

        // Get files from either the HuggingFace API, or from a specified local directory
        let (config_filename, tokenizer_filename, weights_filename) = {
            match &config.model_path {
                Some(base_path) => {
                    if !base_path.is_dir() {
                        bail!("Model path {} is not a directory.", base_path.display());
                    }

                    let config_file = base_path.join("config.json");
                    let tokenizer_file = base_path.join("tokenizer.json");
                    let weights_file = if config.use_pth {
                        base_path.join("pytorch_model.bin")
                    } else {
                        base_path.join("model.safetensors")
                    };
                    (config_file, tokenizer_file, weights_file)
                }
                None => {
                    if config.model_id.is_none() {
                        bail!("Either model_id or model_path must be specified");
                    }

                    let repo = Repo::with_revision(
                        config.model_id.unwrap(),
                        RepoType::Model,
                        config.revision.clone(),
                    );
                    let api = Api::new()?;
                    let api = api.repo(repo);
                    let config_file = api.get("config.json").await?;
                    let tokenizer_file = api.get("tokenizer.json").await?;
                    let weights_file = if config.use_pth {
                        api.get("pytorch_model.bin").await?
                    } else {
                        api.get("model.safetensors").await?
                    };
                    (config_file, tokenizer_file, weights_file)
                }
            }
        };

        let model_config = std::fs::read_to_string(config_filename)?;
        let model_config: DebertaV2Config = serde_json::from_str(&model_config)?;

        // Command-line id2label takes precedence. Otherwise, use model config's id2label.
        let id2label = if let Some(id2label) = config.id2label {
            id2label
        } else if let Some(id2label) = &model_config.id2label {
            id2label.clone()
        } else {
            bail!("Id2Label not found in the model configuration nor specified as a parameter");
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams::default()));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: config.max_sequence_length,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Tokenizer truncation error: {e}"))?;

        let vb = if config.use_pth {
            VarBuilder::from_pth(
                &weights_filename,
                candle_transformers::models::debertav2::DTYPE,
                &device,
            )?
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[weights_filename],
                    candle_transformers::models::debertav2::DTYPE,
                    &device,
                )?
            }
        };

        let vb = vb.set_prefix("deberta");
        let model =
            DebertaV2SeqClassificationModel::load(vb, &model_config, Some(id2label.clone()))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            id2label,
        })
    }
}

#[async_trait]
impl BatchedEngine for DebertaBatchedEngine {
    #[tracing::instrument(skip(self, requests), fields(batch_size = requests.len()))]
    async fn classify_batch(
        &self,
        requests: Vec<ClassificationRequest>,
    ) -> Result<Vec<Result<ClassificationResponse>>> {
        let mut all_texts = Vec::new();
        let mut request_boundaries = Vec::new();
        let mut current_index = 0;

        // Flatten all input texts from all requests
        for request in &requests {
            request_boundaries.push((current_index, current_index + request.input.len()));
            all_texts.extend_from_slice(&request.input);
            current_index += request.input.len();
        }

        // Tokenize all texts in one batch
        let tokenizer_clone = self.tokenizer.clone();
        let (_, input_ids, attention_mask, token_type_ids) =
            tokio::task::spawn_blocking(move || {
                tokenizer_clone
                    .encode_batch(all_texts, true)
                    .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))
                    .map(|encodings| {
                        let mut encoding_stack = Vec::default();
                        let mut attention_mask_stack = Vec::default();
                        let mut token_type_id_stack = Vec::default();

                        for encoding in &encodings {
                            encoding_stack.push(encoding.get_ids().to_vec());
                            attention_mask_stack.push(encoding.get_attention_mask().to_vec());
                            token_type_id_stack.push(encoding.get_type_ids().to_vec());
                        }

                        (
                            encodings,
                            encoding_stack,
                            attention_mask_stack,
                            token_type_id_stack,
                        )
                    })
            })
            .await??;

        // Convert to tensors
        let input_ids_tensors: Result<Vec<_>> = input_ids
            .iter()
            .map(|ids| Tensor::new(ids.as_slice(), &self.device).map_err(anyhow::Error::from))
            .collect();
        let attention_mask_tensors: Result<Vec<_>> = attention_mask
            .iter()
            .map(|mask| Tensor::new(mask.as_slice(), &self.device).map_err(anyhow::Error::from))
            .collect();
        let token_type_ids_tensors: Result<Vec<_>> = token_type_ids
            .iter()
            .map(|types| Tensor::new(types.as_slice(), &self.device).map_err(anyhow::Error::from))
            .collect();

        let input_ids = Tensor::stack(&input_ids_tensors?, 0)?;
        let attention_mask = Tensor::stack(&attention_mask_tensors?, 0)?;
        let token_type_ids = Tensor::stack(&token_type_ids_tensors?, 0)?;

        // Run inference
        let logits = self
            .model
            .forward(&input_ids, Some(token_type_ids), Some(attention_mask))?;
        let predictions = logits.argmax(1)?.to_vec1::<u32>()?;
        let scores = softmax(&logits, 1)?.to_vec2::<f32>()?;

        let mut responses: Vec<Result<ClassificationResponse>> = Vec::new();

        // Split results back into individual responses
        for (req_idx, request) in requests.iter().enumerate() {
            let (start_idx, end_idx) = request_boundaries[req_idx];
            let request_predictions = &predictions[start_idx..end_idx];
            let request_scores = &scores[start_idx..end_idx];

            let data: Vec<ClassificationData> = request_predictions
                .iter()
                .zip(request_scores.iter())
                .enumerate()
                .map(|(index, (&prediction, probs))| {
                    let label = self
                        .id2label
                        .get(&prediction)
                        .cloned()
                        .unwrap_or_else(|| format!("LABEL_{prediction}"));

                    ClassificationData {
                        index,
                        label,
                        probs: probs.iter().map(|&x| x as f64).collect(),
                        num_classes: self.id2label.len(),
                    }
                })
                .collect();

            let usage = Usage {
                prompt_tokens: request.input.iter().map(|s| s.len() as u32 / 4).sum(),
                total_tokens: request.input.iter().map(|s| s.len() as u32 / 4).sum(),
                completion_tokens: 0,
                prompt_tokens_details: None,
            };

            responses.push(Ok(ClassificationResponse {
                id: format!("classify-{}", Uuid::new_v4().simple()),
                object: "list".to_string(),
                created: Utc::now().timestamp(),
                model: request.model.clone(),
                data,
                usage,
            }));
        }

        Ok(responses)
    }
}
