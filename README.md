# Lightweight AI Inference Pipeline

A production-friendly, lightweight Python inference system for text classification and model serving. This project is designed to deliver efficient AI inference with a small footprint, dynamic batching, optional backend flexibility, and observability for real-world deployment scenarios.

The repository contains a modular inference pipeline for sentiment analysis that can run locally via CLI or expose an HTTP API for high-throughput usage.

## Overview

This project demonstrates a clean architecture for deploying a lightweight AI model in a minimal environment:

- Zero-dependency inference using NumPy as the default execution backend
- Optional Scikit-learn and ONNX support when available
- Dynamic request batching for improved throughput under load
- HTTP-based inference API for serving predictions
- Performance metrics and profiling to evaluate inference latency
- Training utilities and model export workflow for sentiment classification

## Key Features

- Fast and lightweight execution
- Modular preprocessor, backend, and postprocessor pipeline design
- Thread-safe dynamic batching for concurrent requests
- Request profiling with throughput and percentile statistics
- Flexible model backends for NumPy, scikit-learn, and ONNX
- REST API for local or embedded deployment
- End-to-end testing for pipeline behavior and correctness

## Repository Structure

The project is organized as a nested package structure derived from the original source archive:

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

### 2. Navigate to the project package

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

Train the sentiment model used by the pipeline:

```bash
python -m inference_pipeline.main train
```

This generates the necessary serialized model artifacts in the `models` directory.

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

Launch the built-in REST API server with batching enabled:

```bash
python -m inference_pipeline.main server --backend numpy --host 0.0.0.0 --port 8000 --max-batch-size 16 --max-wait-ms 50
```

The service exposes endpoints such as:

- `GET /health` — system health check
- `GET /metrics` — performance metrics and latency telemetry
- `POST /predict` — single or batched text inference requests

Example payload:

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

Run a basic concurrent load test against the API:

```bash
python -m inference_pipeline.main benchmark --threads 10 --duration 5 --query-batch-size 1
```

This helps evaluate request throughput and latency performance under parallel traffic.

## Architecture

The project follows a modular pipeline architecture:

- `pipeline/preprocessors/` — transforms raw text into model-ready representations
- `pipeline/backends/` — runs inference using different model execution engines
- `pipeline/postprocessors/` — converts model outputs into human-readable labels
- `pipeline/manager.py` — orchestrates the full inference flow
- `pipeline/batcher.py` — groups requests for efficient processing
- `pipeline/profiler.py` — records latency and throughput metrics
- `api/server.py` — exposes the inference service over HTTP

## Testing

Run the unit tests for pipeline behavior and batching logic:

```bash
python -m unittest discover -s inference_pipeline/tests
```

## Use Cases

This project is suitable for:

- sentiment analysis services
- lightweight model serving in constrained environments
- experimentation with batching and inference optimization
- educational demonstrations of modular ML pipeline design

## License

This project is distributed under the repository's chosen license. Please review the project files for the exact licensing terms.

## Contributing

Contributions are welcome. If you want to extend the pipeline with new backends, preprocessing logic, or API features, open a pull request with a clear description of the improvement.

## Project Summary

The Lightweight AI Inference Pipeline is a practical example of how to build a compact, modular, and high-throughput inference service for text classification. It balances simplicity, performance, and extensibility while keeping the implementation approachable for learning and deployment.

---

Built for lightweight AI inference, high-throughput serving, and practical experimentation.
