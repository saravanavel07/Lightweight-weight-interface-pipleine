from typing import Any
import numpy as np
from ..base import BaseBackend

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXBackend(BaseBackend):
    """
    A backend wrapper for ONNX models running via ONNX Runtime.
    Automatically handles runtime initialization, input/output node binding, and inference.
    """
    def __init__(self, model_path: str):
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. To use the ONNXBackend, "
                "please run 'pip install onnxruntime'."
            )
        self.model_path = model_path
        self.session = None
        self.input_name = ""
        self.output_name = ""
        self.load_model()

    def load_model(self):
        """Initializes the ONNX runtime InferenceSession and fetches node names."""
        self.session = ort.InferenceSession(self.model_path)
        # Fetch the names of input and output nodes
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, model_inputs: Any) -> Any:
        """
        Runs model inference using ONNX Runtime.
        
        Args:
            model_inputs: A NumPy array matching input dimensions of the ONNX model.
            
        Returns:
            The raw model outputs as a NumPy array.
        """
        X = np.asarray(model_inputs, dtype=np.float32)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name], 
            {self.input_name: X}
        )
        return outputs[0]
