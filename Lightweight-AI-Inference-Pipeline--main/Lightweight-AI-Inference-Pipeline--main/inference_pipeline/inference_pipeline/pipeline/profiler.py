import time
from typing import Dict, List, Any
import threading

class Profiler:
    """
    A thread-safe metrics collector and performance profiler.
    Measures processing latencies across different pipeline stages and throughput.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.request_counts = 0
        self.batch_counts = 0
        self.errors_count = 0
        
        # Latency lists (in milliseconds)
        self.preprocess_latencies: List[float] = []
        self.inference_latencies: List[float] = []
        self.postprocess_latencies: List[float] = []
        self.e2e_latencies: List[float] = []
        
        # System start time for overall throughput calculations
        self.start_time = time.time()

    def record(self, preprocess_sec: float, inference_sec: float, postprocess_sec: float, batch_size: int, has_error: bool = False):
        """Records latency metrics for a completed batch request."""
        with self._lock:
            if has_error:
                self.errors_count += batch_size
                return
            
            self.request_counts += batch_size
            self.batch_counts += 1
            
            # Convert to milliseconds per item (amortized) or keep as batch total? 
            # Usually we track batch latency, but let's store total seconds converted to ms
            prep_ms = preprocess_sec * 1000.0
            inf_ms = inference_sec * 1000.0
            post_ms = postprocess_sec * 1000.0
            e2e_ms = prep_ms + inf_ms + post_ms
            
            self.preprocess_latencies.append(prep_ms)
            self.inference_latencies.append(inf_ms)
            self.postprocess_latencies.append(post_ms)
            self.e2e_latencies.append(e2e_ms)

            # Cap list size to avoid infinite memory growth in long-running servers
            if len(self.e2e_latencies) > 10000:
                self.preprocess_latencies = self.preprocess_latencies[-5000:]
                self.inference_latencies = self.inference_latencies[-5000:]
                self.postprocess_latencies = self.postprocess_latencies[-5000:]
                self.e2e_latencies = self.e2e_latencies[-5000:]

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * percentile
        f = int(k)
        c = f + 1
        if c < len(sorted_vals):
            return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)
        else:
            return sorted_vals[f]

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates and returns a dictionary of aggregate metrics."""
        with self._lock:
            uptime = time.time() - self.start_time
            throughput = self.request_counts / uptime if uptime > 0 else 0.0
            
            total_requests = self.request_counts
            total_batches = self.batch_counts
            
            avg_prep = sum(self.preprocess_latencies) / len(self.preprocess_latencies) if self.preprocess_latencies else 0.0
            avg_inf = sum(self.inference_latencies) / len(self.inference_latencies) if self.inference_latencies else 0.0
            avg_post = sum(self.postprocess_latencies) / len(self.postprocess_latencies) if self.postprocess_latencies else 0.0
            avg_e2e = sum(self.e2e_latencies) / len(self.e2e_latencies) if self.e2e_latencies else 0.0

            p50_e2e = self._calculate_percentile(self.e2e_latencies, 0.50)
            p90_e2e = self._calculate_percentile(self.e2e_latencies, 0.90)
            p99_e2e = self._calculate_percentile(self.e2e_latencies, 0.99)
            
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": total_requests,
                "total_batches": total_batches,
                "errors_count": self.errors_count,
                "throughput_req_per_sec": round(throughput, 2),
                "latency_avg_ms": {
                    "preprocessing": round(avg_prep, 3),
                    "inference": round(avg_inf, 3),
                    "postprocessing": round(avg_post, 3),
                    "end_to_end": round(avg_e2e, 3)
                },
                "latency_percentiles_ms": {
                    "p50": round(p50_e2e, 3),
                    "p90": round(p90_e2e, 3),
                    "p99": round(p99_e2e, 3)
                }
            }
