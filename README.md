# Arbiter

Arbiter is a high-performance inference server for DeBERTa-based text classification models. It provides a REST API that accepts text inputs and returns classification predictions with confidence scores. The server implements batched processing to efficiently handle multiple concurrent requests, automatically grouping individual requests into batches for optimal GPU utilization and throughput. This batching approach significantly reduces inference latency when processing multiple texts compared to individual sequential requests, making it practical for production workloads that require real-time text classification at scale.

## Getting Started

### Prerequisites

- Rust 1.88 or later
- CUDA 12.4+ (for GPU support, optional)
- A DeBERTa model (from Hugging Face Hub or local path)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd arbiter
   ```

2. Build the project:
   ```bash
   # For CPU-only inference
   cargo build --release

   # For GPU inference (requires CUDA)
   cargo build --release --features cuda
   ```

### Running the Server

#### Basic Usage

```bash
# Using a Hugging Face model
./target/release/arbiter --model-id microsoft/deberta-v3-base

# Using a local model
./target/release/arbiter --model-path /path/to/local/model
```

#### Configuration Options

- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8000)
- `--batch-size`: Batch size for processing (default: 8)
- `--tick-duration-ms`: Batch processing interval in milliseconds (default: 100)
- `--max-sequence-length`: Maximum input sequence length (default: 512)
- `--cpu-only`: Force CPU-only inference
- `--id2label`: Label mapping in format "0=No Claim,1=Claim"

#### Example API Usage

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/deberta-v3-base",
    "input": ["This is a test sentence", "Another text to classify"]
  }'
```

### Docker

```bash
# CPU version
docker build --target cpu -t arbiter:cpu .

# GPU version
docker build --target gpu -t arbiter:gpu .

# Run with GPU support
docker run --gpus all -p 8000:8000 -e MODEL_ID=microsoft/deberta-v3-base arbiter:gpu
```

### Monitoring

The server exposes Prometheus metrics at `/metrics` for monitoring request throughput, latency, and other operational metrics.