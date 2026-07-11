import unittest
import numpy as np
import time
import os
import json
import threading
from typing import List, Dict, Any

from inference_pipeline.pipeline.base import BasePreprocessor, BaseBackend, BasePostprocessor
from inference_pipeline.pipeline.profiler import Profiler
from inference_pipeline.pipeline.manager import PipelineManager
from inference_pipeline.pipeline.batcher import DynamicBatcher
from inference_pipeline.pipeline.preprocessors.text import SimpleTextPreprocessor
from inference_pipeline.pipeline.backends.numpy_backend import NumpyMLPBackend
from inference_pipeline.pipeline.postprocessors.classification import ClassificationPostprocessor


class DummyPreprocessor(BasePreprocessor):
    def preprocess(self, inputs: List[Any]) -> Any:
        # Convert list of strings to standard representation: string lengths
        return np.array([[len(str(x))] for x in inputs], dtype=np.float32)


class DummyBackend(BaseBackend):
    def predict(self, model_inputs: Any) -> Any:
        # Convert dimensions into class probabilities: [length / 100.0, 1.0 - (length / 100.0)]
        inputs = np.asarray(model_inputs)
        probs = np.zeros((inputs.shape[0], 2), dtype=np.float32)
        for idx in range(inputs.shape[0]):
            val = float(inputs[idx, 0]) / 100.0
            probs[idx, 0] = min(max(val, 0.0), 1.0)
            probs[idx, 1] = 1.0 - probs[idx, 0]
        return probs


class TestPipelineComponents(unittest.TestCase):
    def test_profiler(self):
        profiler = Profiler()
        profiler.record(0.01, 0.05, 0.005, batch_size=2)
        metrics = profiler.get_metrics()
        
        self.assertEqual(metrics["total_requests"], 2)
        self.assertEqual(metrics["total_batches"], 1)
        self.assertEqual(metrics["errors_count"], 0)
        self.assertAlmostEqual(metrics["latency_avg_ms"]["preprocessing"], 10.0)
        self.assertAlmostEqual(metrics["latency_avg_ms"]["inference"], 50.0)
        self.assertAlmostEqual(metrics["latency_avg_ms"]["postprocessing"], 5.0)
        self.assertAlmostEqual(metrics["latency_avg_ms"]["end_to_end"], 65.0)

    def test_pipeline_manager_flow(self):
        preprocessor = DummyPreprocessor()
        backend = DummyBackend()
        postprocessor = ClassificationPostprocessor(labels=["negative", "positive"])
        
        pipeline = PipelineManager(preprocessor, backend, postprocessor)
        results = pipeline.predict(["ok", "super long sentence"])
        
        self.assertEqual(len(results), 2)
        # Check first prediction "ok" (len 2) -> probability class 0 = 2/100 = 0.02, class 1 = 0.98 -> positive
        self.assertEqual(results[0]["label"], "positive")
        self.assertAlmostEqual(results[0]["confidence"], 0.98)
        # Check second prediction "super long sentence" (len 19) -> class 0 = 19/100 = 0.19, class 1 = 0.81 -> positive
        self.assertEqual(results[1]["label"], "positive")
        self.assertAlmostEqual(results[1]["confidence"], 0.81)
        
        # Check that metrics were collected in the profiler
        metrics = pipeline.profiler.get_metrics()
        self.assertEqual(metrics["total_requests"], 2)
        self.assertEqual(metrics["total_batches"], 1)

    def test_dynamic_batcher(self):
        preprocessor = DummyPreprocessor()
        backend = DummyBackend()
        postprocessor = ClassificationPostprocessor(labels=["negative", "positive"])
        pipeline = PipelineManager(preprocessor, backend, postprocessor)
        
        # Set max_wait_ms to 100ms
        batcher = DynamicBatcher(pipeline, max_batch_size=4, max_wait_ms=100.0)
        batcher.start()
        
        try:
            results = []
            threads = []
            
            def client(text):
                res = batcher.predict_single(text)
                results.append((text, res))
                
            # Fire concurrent requests
            inputs = ["short", "sentence", "longer input string"]
            for item in inputs:
                t = threading.Thread(target=client, args=(item,))
                threads.append(t)
                t.start()
                
            for t in threads:
                t.join()
                
            self.assertEqual(len(results), 3)
            # Make sure we got correct labels back per thread
            for text, res in results:
                expected_len = len(text)
                expected_neg_prob = expected_len / 100.0
                expected_pos_prob = 1.0 - expected_neg_prob
                expected_label = "negative" if expected_neg_prob > expected_pos_prob else "positive"
                self.assertEqual(res["label"], expected_label)
                
            # Verify batching metrics
            metrics = pipeline.profiler.get_metrics()
            # Since requests were fired concurrently, they should be processed in exactly 1 batch
            self.assertEqual(metrics["total_batches"], 1)
            self.assertEqual(metrics["total_requests"], 3)
            
        finally:
            batcher.stop()

    def test_numpy_mlp_correctness(self):
        # Retrieve the paths relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "sentiment_mlp.json")
        
        # Check if the model is trained, skip if not
        if not os.path.exists(model_path):
            self.skipTest("Trained sentiment model JSON not found. Run training first.")
            
        preprocessor = SimpleTextPreprocessor.from_config(model_path)
        numpy_backend = NumpyMLPBackend(model_path)
        postprocessor = ClassificationPostprocessor(labels=["negative", "positive"])
        
        pipeline = PipelineManager(preprocessor, numpy_backend, postprocessor)
        
        # Run predictions
        results = pipeline.predict([
            "This is absolutely amazing and perfect!",
            "This is horrible, complete garbage, and broke immediately."
        ])
        
        self.assertEqual(len(results), 2)
        # First review is highly positive
        self.assertEqual(results[0]["label"], "positive")
        self.assertGreater(results[0]["confidence"], 0.70)
        # Second review is highly negative
        self.assertEqual(results[1]["label"], "negative")
        self.assertGreater(results[1]["confidence"], 0.70)


if __name__ == "__main__":
    unittest.main()
