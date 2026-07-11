import time
import queue
import threading
from typing import List, Dict, Any, Tuple, Optional
from .manager import PipelineManager


class DynamicBatcher:
    """
    A thread-safe queueing engine that intercepts incoming single requests and pools them into
    larger batches for the PipelineManager, optimizing hardware utilization.
    
    Runs a worker thread in the background that waits for requests. When a request arrives, it starts
    a timer and collects incoming requests until either 'max_batch_size' is met or 'max_wait_ms' expires.
    """
    def __init__(self, pipeline: PipelineManager, max_batch_size: int = 16, max_wait_ms: float = 50.0):
        self.pipeline = pipeline
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0 # Convert to seconds
        
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the background batching worker thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stops the worker thread."""
        self.running = False
        # Insert a sentinel/poison pill to wake up the queue get() if blocking
        self.queue.put(None)
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def predict_single(self, item: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Submits a single request to the dynamic batcher queue and blocks until the
        result is processed by the background worker thread.
        """
        if not self.running:
            # Fallback to direct pipeline prediction if batcher is stopped
            return self.pipeline.predict([item], [metadata] if metadata else None)[0]

        event = threading.Event()
        result_container: List[Any] = []
        
        # Enqueue the single item along with its signaling event and result container
        self.queue.put((item, metadata, event, result_container))
        
        # Wait for the background worker to signal completion
        # Use a timeout of 10 seconds to avoid permanent blocking if something fails
        if not event.wait(timeout=10.0):
            raise TimeoutError("Inference timed out in the dynamic batcher queue.")
            
        if not result_container:
            raise RuntimeError("Request processed but no result was returned.")
            
        result = result_container[0]
        if isinstance(result, Exception):
            raise result
            
        return result

    def _worker_loop(self):
        """Main loop for the background thread collecting and running batch inference."""
        while self.running:
            try:
                # 1. Block until the first item arrives in the queue
                first_item_tuple = self.queue.get()
                if first_item_tuple is None:  # Stop signal received
                    break
                
                batch_items = [first_item_tuple]
                start_time = time.time()
                
                # 2. Collect additional items until max_batch_size or max_wait_ms expires
                while len(batch_items) < self.max_batch_size:
                    elapsed = time.time() - start_time
                    remaining = self.max_wait_ms - elapsed
                    
                    if remaining <= 0:
                        break
                        
                    try:
                        # Wait for next item with timeout
                        next_item = self.queue.get(timeout=remaining)
                        if next_item is None:
                            # Poison pill, stop immediately but process whatever we have first
                            self.running = False
                            break
                        batch_items.append(next_item)
                    except queue.Empty:
                        break  # Timeout reached, run inference on what we have

                # 3. Separate collected batch elements
                # batch_items format: List[Tuple[item, metadata, event, result_container]]
                raw_inputs = [x[0] for x in batch_items]
                metadata_list = [x[1] for x in batch_items]
                events = [x[2] for x in batch_items]
                result_containers = [x[3] for x in batch_items]
                
                # 4. Run batch prediction
                try:
                    # Provide metadata list if any item has metadata
                    has_metadata = any(m is not None for m in metadata_list)
                    meta_arg = metadata_list if has_metadata else None
                    
                    batch_results = self.pipeline.predict(raw_inputs, meta_arg)
                    
                    # Distribute results to their respective containers
                    for idx, result in enumerate(batch_results):
                        result_containers[idx].append(result)
                except Exception as e:
                    # Distribute the exception to all callers if execution failed
                    for container in result_containers:
                        container.append(e)
                finally:
                    # 5. Signal all waiting threads to resume
                    for event in events:
                        event.set()
                        
            except Exception as e:
                # Catch unexpected failures in worker loop to prevent thread crash
                print(f"[DynamicBatcher] Worker loop error: {e}")
                time.sleep(0.01)
