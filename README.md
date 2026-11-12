# Lightweight AI Inference Pipeline

A production-friendly, lightweight Python project for text classification and AI inference serving. This repository provides a clean pipeline for sentiment analysis with modular preprocessing, backend inference, postprocessing, dynamic batching, and a simple HTTP API.

The project is designed to run efficiently in small environments while still supporting modern inference workflows and benchmarking.

## Live Demo

You can run the project locally as a live inference demo:

```bash
cd Lightweight-AI-Inference-Pipeline--main/Lightweight-AI-Inference-Pipeline--main/inference_pipeline
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m inference_pipeline.main server --backend numpy --host 0.0.0.0 --port 8000
```

Then test it with:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input":"I love this product!"}'
```

You can also open these endpoints in the browser:

- http://localhost:8000/health
- http://localhost:8000/metrics

If you deploy this project on a cloud platform, replace localhost with your public server URL.

## Overview

This repository demonstrates how to build a compact AI inference system with:

- Zero-dependency NumPy-based inference by default
- Optional scikit-learn and ONNX support
- Dynamic request batching for throughput optimization
- REST API for serving predictions
- Metrics and latency profiling
- Training and export utilities for the model

## Key Features

- Lightweight and fast inference runtime
- Modular pipeline architecture
- Thread-safe dynamic batching
- Request profiling and throughput telemetry
- Configurable model backends
- Local CLI and HTTP API execution
- Unit-tested pipeline behavior

## Repository Structure

```text
Lightweight-weight-interface-pipleine/
├── README.md
└── Lightweight-AI-Inference-Pipeline--main/
    └── Lightweight-AI-Inference-Pipeline--main/
        └── inference_pipeline/
            ├── inference_pipeline/
            │   ├── api/
            │   ├── models/
            │   ├── pipeline/
            │   ├── tests/
            │   ├── main.py
            │   ├── README.md
            │   └── requirements.txt
            └── inference_pipeline.zip
```

## Tech Stack

- Python 3.x
- NumPy
- scikit-learn (optional)
- ONNX Runtime (optional)
- Standard library HTTP server

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/saravanavel07/Lightweight-weight-interface-pipleine.git
cd Lightweight-weight-interface-pipleine
```

### 2. Navigate to the project directory

```bash
cd Lightweight-AI-Inference-Pipeline--main/Lightweight-AI-Inference-Pipeline--main/inference_pipeline
```

### 3. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Training the Model

```bash
python -m inference_pipeline.main train
```

This creates the export files and model artifacts used by the pipeline.

## Running Predictions

### Offline prediction with NumPy backend

```bash
python -m inference_pipeline.main predict "This product is amazing and very fast!" --backend numpy
```

### Offline prediction with scikit-learn backend

```bash
python -m inference_pipeline.main predict "This product is terrible and unreliable." --backend sklearn
```

## Starting the API Server

```bash
python -m inference_pipeline.main server --backend numpy --host 0.0.0.0 --port 8000 --max-batch-size 16 --max-wait-ms 50
```

Available endpoints:

- `GET /health` — health status
- `GET /metrics` — performance and latency metrics
- `POST /predict` — single or batched inference requests

Example input payload:

```json
{
  "input": "I love this product!"
}
```

Batch payload:

```json
{
  "inputs": ["Amazing quality", "Terrible experience", "Very slow service"]
}
```

## Benchmarking

```bash
python -m inference_pipeline.main benchmark --threads 10 --duration 5 --query-batch-size 1
```

This helps measure throughput, latency, and batching efficiency under concurrent usage.

## Architecture

The project follows a modular design:

- `pipeline/preprocessors/` — text-to-feature preprocessing
- `pipeline/backends/` — model inference engines
- `pipeline/postprocessors/` — label generation and output formatting
- `pipeline/manager.py` — orchestration logic
- `pipeline/batcher.py` — request batching
- `pipeline/profiler.py` — profiling and metrics
- `api/server.py` — HTTP API server

## Testing

```bash
python -m unittest discover -s inference_pipeline/tests
```

## Use Cases

- Sentiment analysis services
- Lightweight AI inference in resource-constrained environments
- Model serving experiments and benchmarking
- Educational demos of modular ML pipelines

## License

This project is distributed under the repository's chosen license. Please review the project files for the exact licensing terms.

## Contributing

Contributions are welcome. If you want to add a new backend, improve preprocessing, or extend the API, open a pull request with a clear explanation of the change.

## Project Summary

The Lightweight AI Inference Pipeline is a compact, modular, and efficient solution for text classification and model serving. It combines simplicity, performance, and extensibility to make lightweight AI deployment straightforward.

---

Built for lightweight AI inference, high-throughput serving, and practical experimentation.
