import time
from typing import List, Dict, Any, Optional
from .base import BasePreprocessor, BaseBackend, BasePostprocessor
from .profiler import Profiler


class PipelineManager:
    """
    Orchestrator that integrates a preprocessor, a model backend, and a postprocessor.
    Executes end-to-end inference while gathering profiling telemetry.
    """
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        backend: BaseBackend,
        postprocessor: BasePostprocessor,
        profiler: Optional[Profiler] = None
    ):
        self.preprocessor = preprocessor
        self.backend = backend
        self.postprocessor = postprocessor
        self.profiler = profiler if profiler is not None else Profiler()

    def predict(self, raw_inputs: List[Any], metadata: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """
        Executes the end-to-end inference pipeline on a batch of raw inputs.
        
        Args:
            raw_inputs: List of input data items (e.g. strings, dictionaries, numerical vectors).
            metadata: Optional list of dictionaries containing request metadata.
            
        Returns:
            List of postprocessed prediction dictionaries.
        """
        if not raw_inputs:
            return []
            
        batch_size = len(raw_inputs)
        has_error = False
        
        t0 = time.perf_counter()
        
        # 1. Preprocessing Stage
        try:
            model_inputs = self.preprocessor.preprocess(raw_inputs)
            t1 = time.perf_counter()
        except Exception as e:
            # If preprocessing fails, record failure and re-raise
            self.profiler.record(0.0, 0.0, 0.0, batch_size, has_error=True)
            raise RuntimeError(f"Preprocessing failed: {e}") from e

        # 2. Inference Stage
        try:
            model_outputs = self.backend.predict(model_inputs)
            t2 = time.perf_counter()
        except Exception as e:
            # If inference fails, record failure and re-raise
            self.profiler.record(t1 - t0, 0.0, 0.0, batch_size, has_error=True)
            raise RuntimeError(f"Model inference failed: {e}") from e

        # 3. Postprocessing Stage
        try:
            results = self.postprocessor.postprocess(model_outputs, metadata)
            t3 = time.perf_counter()
        except Exception as e:
            # If postprocessing fails, record failure and re-raise
            self.profiler.record(t1 - t0, t2 - t1, 0.0, batch_size, has_error=True)
            raise RuntimeError(f"Postprocessing failed: {e}") from e

        # Calculate time spent on each step
        preprocess_sec = t1 - t0
        inference_sec = t2 - t1
        postprocess_sec = t3 - t2
        
        # Record metrics in the profiler
        self.profiler.record(preprocess_sec, inference_sec, postprocess_sec, batch_size)
        
        return results
