# Lightweight AI Inference Pipeline

A high-performance, modular, and **zero-dependency out-of-the-box** Python library for AI model serving. It features dynamic batching, a performance profiling engine, multi-backend execution, and a built-in concurrent REST API server.

---

## Key Features

- 🚀 **Zero-Dependency by Default**: Runs entirely on Python's standard library and `numpy`. Scikit-learn and ONNX Runtime are supported dynamically if installed.
- ⚡ **Dynamic Batching**: Intercepts single inference requests and pools them into optimal batch sizes within a latency timeout window. Improves server throughput under concurrent load.
- 📈 **High-Resolution Profiling**: Tracks preprocessing, model execution, and postprocessing latency, system errors, throughput, and $P_{50}, P_{90}, P_{99}$ latencies.
- 🌐 **Concurrent HTTP REST Server**: Zero-dependency `http.server`-based server using thread pool dispatching for concurrent inference.
- 🛠️ **Benchmarking Utility**: A built-in load-testing tool to measure performance metrics under parallel threads.

---

## Directory Layout

```
inference_pipeline/
│
├── README.md                   # Project Documentation
├── requirements.txt            # Package Dependencies
│
├── pipeline/
│   ├── base.py                 # Abstract Base Protocols (Preprocessor, Backend, Postprocessor)
│   ├── manager.py              # PipelineManager (Orchestrates stages + tracks telemetry)
│   ├── batcher.py              # DynamicBatcher (Thread-safe request queueing)
│   ├── profiler.py             # Performance Profiler & Metrics Accumulator
│   │
│   ├── backends/
│   │   ├── numpy_backend.py    # Zero-dependency NumPy MLP forward-pass
│   │   ├── sklearn_backend.py  # Scikit-learn Backend wrapper
│   │   └── onnx_backend.py     # ONNX Runtime Backend wrapper
│   │
│   ├── preprocessors/
│   │   └── text.py             # Pure Python/NumPy TF-IDF text vectorizer
│   │
│   └── postprocessors/
│       └── classification.py   # Logits-to-labels postprocessor with softmax/thresholding
│
├── models/
│   ├── train_sentiment.py      # Trains a sentiment model and exports weights & pickled models
│   └── sentiment_mlp.json      # Trained weights for NumPy/Sklearn backend
│
├── api/
│   └── server.py               # Built-in multi-threaded HTTP REST Server
│
├── tests/
│   └── test_pipeline.py        # Automated test suite (All components, batcher & correctness)
│
└── main.py                     # CLI Entrypoint (Train, Predict, Server, Benchmark)
```

---

## Quick Start Guide

All commands should be executed from the project root directory containing `inference_pipeline`.

### 1. Installation
Install the optional dependencies (`scikit-learn` and `numpy`):
```bash
pip install -r requirements.txt
```

### 2. Model Training & Export
Train a Multilayer Perceptron (MLP) model for sentiment classification. This generates the Scikit-learn pickle file and extracts the weights/biases into a JSON format for the zero-dependency NumPy backend:
```bash
python -m inference_pipeline.main train
```

### 3. Local CLI Prediction
Run offline predictions via the CLI using either the pure `numpy` or `sklearn` backend:
```bash
# Using the zero-dependency NumPy MLP Backend (Default)
python -m inference_pipeline.main predict "This product is fantastic!" "Awful customer service, very slow." --backend numpy

# Using the Scikit-learn Backend
python -m inference_pipeline.main predict "Incredible quality!" "Do not buy this." --backend sklearn
```

### 4. Run the REST API Server
Start the concurrent HTTP REST API server with Dynamic Batching enabled (polls incoming single queries, batches them, runs inference, and responds):
```bash
python -m inference_pipeline.main server --backend numpy --port 8000 --max-batch-size 8 --max-wait-ms 50
```

#### API Endpoints
- **Health Check (`GET /health`)**:
  ```json
  {
    "status": "healthy",
    "timestamp": 1783849182.1,
    "batcher_running": true,
    "dynamic_batching_enabled": true
  }
  ```
- **Telemetry Metrics (`GET /metrics`)**:
  Returns detailed execution profiling stats:
  ```json
  {
    "uptime_seconds": 12.5,
    "total_requests": 240,
    "total_batches": 30,
    "throughput_req_per_sec": 19.2,
    "latency_avg_ms": {
      "preprocessing": 0.42,
      "inference": 1.25,
      "postprocessing": 0.12,
      "end_to_end": 1.79
    },
    "latency_percentiles_ms": {
      "p50": 1.52,
      "p90": 2.11,
      "p99": 3.42
    }
  }
  ```
- **Prediction Input (`POST /predict`)**:
  Single Query:
  ```json
  {
    "input": "I love the new layout!"
  }
  ```
  Batch Query:
  ```json
  {
    "inputs": ["It is amazing!", "This crashes instantly."]
  }
  ```

### 5. Benchmark under Concurrent Load
With the API server running in a separate terminal, launch a multi-threaded load test to benchmark performance and observe the benefits of dynamic batching:
```bash
python -m inference_pipeline.main benchmark --threads 10 --duration 5 --query-batch-size 1
```

---

## Technical Details

### Pure NumPy MLP Forward Pass
The NumPy backend (`pipeline/backends/numpy_backend.py`) implements a standard dense forward pass:
$$z^{[1]} = X W^{[1]} + b^{[1]}$$
$$a^{[1]} = \text{ReLU}(z^{[1]})$$
$$z^{[2]} = a^{[1]} W^{[2]} + b^{[2]}$$
$$a^{[2]} = \text{Softmax}(z^{[2]})$$
This allows the engine to run without any massive framework installs like PyTorch or TensorFlow, making it highly portable.

### Dynamic Batching Mechanics
When multiple parallel client requests query the server, the `DynamicBatcher` buffers them in a thread-safe Queue. The background coordinator worker wakes up, gathers inputs up to the `max_batch_size` or waits until the timer `max_wait_ms` has ticked, compiles a single batch array, runs the inference once, and returns individual results using client synchronization (`threading.Event`).
