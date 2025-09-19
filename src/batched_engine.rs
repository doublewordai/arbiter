use anyhow::Result;
use async_trait::async_trait;
use std::collections::VecDeque;
use tokio::sync::oneshot;
use tokio::time::{Instant, interval};

use crate::config::BatchConfig;
use crate::engine::BatchedEngine;
use crate::engine::Engine;
use crate::types::{ClassificationRequest, ClassificationResponse};

type ResponseSender = oneshot::Sender<Result<ClassificationResponse>>;

#[derive(Debug)]
struct QueuedRequest {
    request: ClassificationRequest,
    response_tx: ResponseSender,
}

pub struct BatchedEngineWrapper {
    request_tx: flume::Sender<QueuedRequest>,
}

impl BatchedEngineWrapper {
    pub fn new<T: BatchedEngine + 'static>(
        config: BatchConfig,
        batched_engine: T,
    ) -> (Self, BatchProcessor<T>) {
        let (request_tx, request_rx) = flume::bounded(0); // Rendezvous channel

        let processor = BatchProcessor {
            request_rx,
            config,
            request_queue: VecDeque::new(),
            batched_engine,
        };

        let engine = Self { request_tx };

        (engine, processor)
    }
}

#[async_trait]
impl Engine for BatchedEngineWrapper {
    #[tracing::instrument(skip(self, request), fields(input_count = request.input.len()))]
    async fn classify(&self, request: ClassificationRequest) -> Result<ClassificationResponse> {
        let (response_tx, response_rx) = oneshot::channel();

        let queued_request = QueuedRequest {
            request,
            response_tx,
        };

        self.request_tx
            .send_async(queued_request)
            .await
            .map_err(|_| anyhow::anyhow!("Engine queue is closed"))?;

        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Response channel closed"))?
    }
}

pub struct BatchProcessor<T: BatchedEngine> {
    request_rx: flume::Receiver<QueuedRequest>,
    config: BatchConfig,
    request_queue: VecDeque<QueuedRequest>,
    batched_engine: T,
}

impl<T: BatchedEngine> BatchProcessor<T> {
    #[tracing::instrument(skip(self))]
    pub async fn run_forever(mut self) -> Result<()> {
        let mut tick_timer = interval(self.config.tick_duration);

        loop {
            tokio::select! {
                // Receive new requests
                request = self.request_rx.recv_async() => {
                    match request {
                        Ok(req) => {
                            self.request_queue.push_back(req);
                            tracing::debug!(queue_size = self.request_queue.len(), "Request received and queued");

                            // If we have enough requests, process a batch immediately
                            if self.request_queue.len() >= self.config.batch_size {
                                tracing::debug!(batch_size = self.config.batch_size, "Batch size reached, processing immediately");
                                self.process_batch().await;
                            }
                        }
                        Err(_) => {
                            tracing::info!("Channel closed, processing remaining requests and exiting");
                            // Channel closed, process remaining requests and exit
                            if !self.request_queue.is_empty() {
                                self.process_batch().await;
                            }
                            break Ok(());
                        }
                    }
                }

                // Tick timer - process pending requests even if batch isn't full
                _ = tick_timer.tick() => {
                    if !self.request_queue.is_empty() {
                        tracing::debug!(pending_requests = self.request_queue.len(), "Tick timer fired, processing pending requests");
                        self.process_batch().await;
                    } else {
                        tracing::trace!("Tick timer fired but no pending requests");
                    }
                }
            }
        }
    }

    #[tracing::instrument(skip(self))]
    async fn process_batch(&mut self) {
        let batch_start = Instant::now();

        // Take up to batch_size requests in FIFO order
        let batch: Vec<_> = self
            .request_queue
            .drain(..self.config.batch_size.min(self.request_queue.len()))
            .collect();

        if batch.is_empty() {
            return;
        }

        tracing::info!(batch_size = batch.len(), "Processing batch");

        // Extract requests and response channels
        let requests: Vec<_> = batch.iter().map(|req| req.request.clone()).collect();
        let response_channels: Vec<_> = batch.into_iter().map(|req| req.response_tx).collect();

        // Process batch through the batched engine
        tracing::debug!("Calling classify_batch on engine");
        let responses = self.batched_engine.classify_batch(requests).await;

        // Send responses back
        match responses {
            Ok(response_vec) => {
                tracing::debug!(
                    response_count = response_vec.len(),
                    "Batch processing successful"
                );
                for (response_tx, response_result) in
                    response_channels.into_iter().zip(response_vec.into_iter())
                {
                    let _ = response_tx.send(response_result);
                }
            }
            Err(err) => {
                tracing::error!("Batch processing failed: {}", err);
                // Send error to all pending requests
                for response_tx in response_channels {
                    let _ =
                        response_tx.send(Err(anyhow::anyhow!("Batch processing failed: {}", err)));
                }
            }
        }

        let processing_time = batch_start.elapsed();
        tracing::info!(
            processing_time_ms = processing_time.as_millis(),
            "Batch processed"
        );
    }
}
