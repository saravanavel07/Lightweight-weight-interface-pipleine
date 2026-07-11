from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

class BasePreprocessor(ABC):
    """
    Abstract base class for all input preprocessors.
    Converts raw incoming request formats into model-compatible inputs.
    """
    @abstractmethod
    def preprocess(self, inputs: List[Any]) -> Any:
        """
        Preprocesses a batch of raw inputs.
        
        Args:
            inputs: A list of raw input data items.
            
        Returns:
            Features formatted correctly for the backend model.
        """
        pass


class BaseBackend(ABC):
    """
    Abstract base class for all model backends (NumPy, Scikit-learn, ONNX, etc.).
    Handles the actual execution/forward pass of the model.
    """
    @abstractmethod
    def predict(self, model_inputs: Any) -> Any:
        """
        Executes model inference on preprocessed features.
        
        Args:
            model_inputs: Preprocessed features.
            
        Returns:
            Raw model outputs (logits, probabilities, array predictions, etc.).
        """
        pass


class BasePostprocessor(ABC):
    """
    Abstract base class for all output postprocessors.
    Translates raw model output vectors into structured, readable predictions.
    """
    @abstractmethod
    def postprocess(self, model_outputs: Any, metadata: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """
        Postprocesses model outputs.
        
        Args:
            model_outputs: Raw outputs from the model backend.
            metadata: Optional list of dictionaries with context/metadata per input.
            
        Returns:
            A list of postprocessed predictions.
        """
        pass
