import json
import time
import urllib.parse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingTCPServer
from typing import Any, Dict, List, Optional
import traceback

from ..pipeline.manager import PipelineManager
from ..pipeline.batcher import DynamicBatcher


# Class variable to hold active components so the RequestHandler can access them
class ServerState:
    pipeline: Optional[PipelineManager] = None
    batcher: Optional[DynamicBatcher] = None
    use_batching: bool = False


class PipelineHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Request handler serving REST endpoints for health checks, metrics, and inference.
    Extends BaseHTTPRequestHandler to handle POST and GET requests.
    """
    def _send_response_json(self, status_code: int, data: Dict[str, Any]):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handles GET requests for /health and /metrics."""
        parsed_url = urllib.parse.urlparse(self.path)
        
        if parsed_url.path == "/health":
            self._send_response_json(200, {
                "status": "healthy",
                "timestamp": time.time(),
                "batcher_running": ServerState.batcher.running if ServerState.batcher else False,
                "dynamic_batching_enabled": ServerState.use_batching
            })
            
        elif parsed_url.path == "/metrics":
            if ServerState.pipeline and ServerState.pipeline.profiler:
                metrics = ServerState.pipeline.profiler.get_metrics()
                # Inject additional server-level metadata
                metrics["dynamic_batching_enabled"] = ServerState.use_batching
                if ServerState.batcher:
                    metrics["batcher_queue_size"] = ServerState.batcher.queue.qsize()
                self._send_response_json(200, metrics)
            else:
                self._send_response_json(500, {"error": "Profiler not initialized."})
                
        else:
            self._send_response_json(404, {"error": "Endpoint not found. Use /health or /metrics."})

    def do_POST(self):
        """Handles POST requests for /predict."""
        parsed_url = urllib.parse.urlparse(self.path)
        
        if parsed_url.path != "/predict":
            self._send_response_json(404, {"error": "Endpoint not found. Use POST /predict."})
            return
            
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_response_json(400, {"error": "Empty request body."})
                return
                
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body)
            
            # Format 1: Batch inputs e.g. {"inputs": ["text 1", "text 2"]}
            # Format 2: Single input e.g. {"input": "text 1"}
            inputs: List[Any] = []
            is_batch_request = False
            
            if "inputs" in payload:
                if not isinstance(payload["inputs"], list):
                    self._send_response_json(400, {"error": "'inputs' field must be a JSON array."})
                    return
                inputs = payload["inputs"]
                is_batch_request = True
            elif "input" in payload:
                inputs = [payload["input"]]
            else:
                self._send_response_json(400, {"error": "Payload must contain either 'input' (string/object) or 'inputs' (list)."})
                return
                
            if not inputs:
                self._send_response_json(400, {"error": "Inputs list cannot be empty."})
                return

            if ServerState.pipeline is None:
                self._send_response_json(500, {"error": "Pipeline model is not loaded on the server."})
                return

            results = []
            
            # Execute prediction
            if ServerState.use_batching and ServerState.batcher:
                # Dynamic batching route (processes items concurrently through the queue)
                # To handle a batch request via dynamic batcher, we enqueue them individually and gather results
                threads = []
                results_collected = [None] * len(inputs)
                
                def predict_worker(idx, item):
                    try:
                        results_collected[idx] = ServerState.batcher.predict_single(item)
                    except Exception as exc:
                        results_collected[idx] = {"error": str(exc)}
                
                for idx, item in enumerate(inputs):
                    t = threading.Thread(target=predict_worker, args=(idx, item))
                    threads.append(t)
                    t.start()
                    
                for t in threads:
                    t.join()
                
                results = results_collected
            else:
                # Direct prediction route (synchronous execution on the main pipeline thread)
                results = ServerState.pipeline.predict(inputs)
                
            response_data = {
                "results": results if is_batch_request else results[0],
                "batching_applied": ServerState.use_batching,
                "count": len(inputs)
            }
            self._send_response_json(200, response_data)
            
        except json.JSONDecodeError:
            self._send_response_json(400, {"error": "Invalid JSON format in body."})
        except Exception as e:
            traceback.print_exc()
            self._send_response_json(500, {"error": f"Internal server error: {str(e)}"})


class ConcurrentHTTPServer(ThreadingTCPServer):
    """
    Threaded TCP server that handles each connection request in a separate thread.
    This guarantees overlapping/concurrent calls, enabling dynamic batching to work.
    """
    allow_reuse_address = True


def start_server(
    pipeline: PipelineManager,
    host: str = "0.0.0.0",
    port: int = 8000,
    use_batching: bool = True,
    max_batch_size: int = 16,
    max_wait_ms: float = 50.0
):
    """Initializes ServerState and starts the threaded HTTP inference server."""
    import threading
    
    # Store state globally for the RequestHandler classes
    ServerState.pipeline = pipeline
    ServerState.use_batching = use_batching
    
    if use_batching:
        print(f"[Server] Enabling Dynamic Batching (max_size={max_batch_size}, wait={max_wait_ms}ms)")
        ServerState.batcher = DynamicBatcher(pipeline, max_batch_size, max_wait_ms)
        ServerState.batcher.start()
    else:
        print("[Server] Dynamic Batching is DISABLED. Requests are routed directly.")

    server_address = (host, port)
    httpd = ConcurrentHTTPServer(server_address, PipelineHTTPRequestHandler)
    print(f"[Server] Inference API listening at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down inference server...")
    finally:
        if ServerState.batcher:
            ServerState.batcher.stop()
        httpd.server_close()
        print("[Server] Server stopped.")
