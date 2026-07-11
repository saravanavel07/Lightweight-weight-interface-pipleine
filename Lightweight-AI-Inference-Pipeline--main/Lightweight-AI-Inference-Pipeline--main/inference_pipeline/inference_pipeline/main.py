import argparse
import sys
import os
import json
import time
import threading
import urllib.request
from typing import List

# Add parent directory to path to ensure proper module imports when running main.py directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_pipeline.pipeline.manager import PipelineManager
from inference_pipeline.pipeline.preprocessors.text import SimpleTextPreprocessor
from inference_pipeline.pipeline.backends.numpy_backend import NumpyMLPBackend
from inference_pipeline.pipeline.backends.sklearn_backend import SklearnBackend
from inference_pipeline.pipeline.postprocessors.classification import ClassificationPostprocessor
from inference_pipeline.api.server import start_server


def setup_numpy_pipeline(model_path: str) -> PipelineManager:
    """Helper to initialize the NumPy-based pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model parameters JSON not found at {model_path}. Please run training first.")
    
    preprocessor = SimpleTextPreprocessor.from_config(model_path)
    backend = NumpyMLPBackend(model_path)
    
    # Load labels list from model JSON
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    labels = model_data.get("labels", ["negative", "positive"])
    
    postprocessor = ClassificationPostprocessor(labels=labels)
    return PipelineManager(preprocessor, backend, postprocessor)


def setup_sklearn_pipeline(model_path: str, config_path: str) -> PipelineManager:
    """Helper to initialize the Scikit-learn based pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pickled model not found at {model_path}. Please run training first.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config JSON (for vocabulary) not found at {config_path}.")
        
    preprocessor = SimpleTextPreprocessor.from_config(config_path)
    backend = SklearnBackend(model_path)
    
    with open(config_path, 'r') as f:
        model_data = json.load(f)
    labels = model_data.get("labels", ["negative", "positive"])
    
    postprocessor = ClassificationPostprocessor(labels=labels)
    return PipelineManager(preprocessor, backend, postprocessor)


def run_prediction(args):
    """Executes a single prediction query from CLI arguments."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if args.backend == "numpy":
            model_path = os.path.join(base_dir, "models", "sentiment_mlp.json")
            pipeline = setup_numpy_pipeline(model_path)
        else:
            model_path = os.path.join(base_dir, "models", "sentiment_model.pkl")
            config_path = os.path.join(base_dir, "models", "sentiment_mlp.json")
            pipeline = setup_sklearn_pipeline(model_path, config_path)
            
        print(f"Running inference on {len(args.inputs)} inputs using {args.backend} backend...")
        results = pipeline.predict(args.inputs)
        
        for inp, res in zip(args.inputs, results):
            print(f"\nInput: '{inp}'")
            print(f"Prediction: {res['label']} (Confidence: {res['confidence']:.4f})")
            print(f"Distribution: {res['probabilities']}")
            
    except Exception as e:
        print(f"Error executing prediction: {e}")
        sys.exit(1)


def run_training():
    """Triggers the model training script."""
    print("Launching training script...")
    from inference_pipeline.models.train_sentiment import main as train_main
    train_main()


def start_api_server(args):
    """Starts the REST server."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if args.backend == "numpy":
            model_path = os.path.join(base_dir, "models", "sentiment_mlp.json")
            pipeline = setup_numpy_pipeline(model_path)
        else:
            model_path = os.path.join(base_dir, "models", "sentiment_model.pkl")
            config_path = os.path.join(base_dir, "models", "sentiment_mlp.json")
            pipeline = setup_sklearn_pipeline(model_path, config_path)
            
        start_server(
            pipeline=pipeline,
            host=args.host,
            port=args.port,
            use_batching=args.batching,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)


def run_benchmark(args):
    """
    Runs a multithreaded benchmarking client against a running HTTP server
    to test throughput, latency metrics, and performance behavior under load.
    """
    url = f"http://{args.host}:{args.port}/predict"
    print(f"Starting benchmark against {url}")
    print(f"Configuration: Threads={args.threads}, Duration={args.duration}s, BatchSize={args.query_batch_size}")
    
    sentences = [
        "This is an absolutely amazing product, highly recommended!",
        "I love this so much! It works perfectly and is very fast.",
        "This is terrible. It broke after just one day of usage.",
        "I hate it. It is extremely slow, laggy, and useless.",
        "Perfect solution for my needs. Fast, reliable, and lightweight.",
        "Great design, extremely useful, and worth every single penny.",
        "Do not buy this! It is a complete waste of time and money.",
        "Very happy with this product. Exceeded my high expectations.",
        "Horrible customer support and poor product build quality.",
        "Fails to load, crashes on startup, and is totally broken."
    ]
    
    # Pre-construct requests payload
    if args.query_batch_size == 1:
        payloads = [{"input": s} for s in sentences]
    else:
        # Create batches of sentences
        payloads = []
        for i in range(0, len(sentences), args.query_batch_size):
            chunk = sentences[i:i + args.query_batch_size]
            if chunk:
                payloads.append({"inputs": chunk})

    results = []
    stop_event = threading.Event()
    
    def worker():
        import random
        while not stop_event.is_set():
            payload = random.choice(payloads)
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url, 
                data=data, 
                headers={'Content-Type': 'application/json'}
            )
            
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    res_body = response.read()
                    status = response.status
                    t1 = time.perf_counter()
                    
                    if status == 200:
                        results.append((t1 - t0, True, args.query_batch_size))
                    else:
                        results.append((t1 - t0, False, args.query_batch_size))
            except Exception as e:
                t1 = time.perf_counter()
                results.append((t1 - t0, False, args.query_batch_size))

    # Spawn worker threads
    threads = []
    for _ in range(args.threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    time.sleep(args.duration)
    stop_event.set()
    
    for t in threads:
        t.join()
        
    if not results:
        print("No requests completed.")
        return
        
    # Analyze results
    latencies = [r[0] * 1000.0 for r in results if r[1]] # ms
    failures = len([r for r in results if not r[1]])
    successes = len(latencies)
    total_reqs = len(results)
    
    # Calculate sum of items predicted (multiply by query_batch_size)
    total_predictions = sum(r[2] for r in results if r[1])
    
    uptime = args.duration
    throughput_req = successes / uptime
    throughput_pred = total_predictions / uptime
    
    latencies.sort()
    
    def get_percentile(sorted_list, pct):
        if not sorted_list:
            return 0.0
        idx = int(len(sorted_list) * pct)
        if idx >= len(sorted_list):
            idx = len(sorted_list) - 1
        return sorted_list[idx]
        
    p50 = get_percentile(latencies, 0.50)
    p90 = get_percentile(latencies, 0.90)
    p99 = get_percentile(latencies, 0.99)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    
    # Print clean results table
    print("\n" + "=" * 40)
    print("          BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Total Requests Sent:   {total_reqs}")
    print(f"Successful Requests:  {successes}")
    print(f"Failed Requests:      {failures}")
    print(f"Success Rate:         {(successes/total_reqs)*100:.2f}%" if total_reqs else "0.00%")
    print("-" * 40)
    print(f"Throughput (HTTP req): {throughput_req:.2f} req/sec")
    print(f"Throughput (Inferences): {throughput_pred:.2f} inferences/sec")
    print("-" * 40)
    print(f"Avg Latency:           {avg_lat:.2f} ms")
    print(f"P50 Latency (median):  {p50:.2f} ms")
    print(f"P90 Latency (p90):     {p90:.2f} ms")
    print(f"P99 Latency (p99):     {p99:.2f} ms")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Lightweight AI Inference Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Train command
    subparsers.add_parser("train", help="Train and serialize the sentiment analysis model")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Execute offline prediction on text input")
    predict_parser.add_argument("inputs", nargs="+", help="One or more text inputs to evaluate")
    predict_parser.add_argument("--backend", choices=["numpy", "sklearn"], default="numpy", help="Inference backend engine (default: numpy)")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Launch the REST inference API server")
    server_parser.add_argument("--backend", choices=["numpy", "sklearn"], default="numpy", help="Inference backend engine (default: numpy)")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to (default: 0.0.0.0)")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    server_parser.add_argument("--no-batching", action="store_false", dest="batching", help="Disable dynamic batching")
    server_parser.add_argument("--max-batch-size", type=int, default=16, help="Dynamic batcher maximum size")
    server_parser.add_argument("--max-wait-ms", type=float, default=50.0, help="Dynamic batcher wait time in ms")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run client-side performance benchmark load test")
    bench_parser.add_argument("--host", default="localhost", help="Target API server host")
    bench_parser.add_argument("--port", type=int, default=8000, help="Target API server port")
    bench_parser.add_argument("--threads", type=int, default=5, help="Number of concurrent client worker threads")
    bench_parser.add_argument("--duration", type=int, default=5, help="Benchmark run duration in seconds")
    bench_parser.add_argument("--query-batch-size", type=int, default=1, help="Number of input items sent per HTTP request")

    args = parser.parse_args()
    
    if args.command == "train":
        run_training()
    elif args.command == "predict":
        run_prediction(args)
    elif args.command == "server":
        start_api_server(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
