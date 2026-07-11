import pickle
from typing import Any
from ..base import BaseBackend

class SklearnBackend(BaseBackend):
    """
    A backend wrapper for Scikit-Learn models serialized using pickle.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads a pickled scikit-learn model."""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, model_inputs: Any) -> Any:
        """
        Runs prediction. Prefers predict_proba if available, falling back to predict.
        
        Args:
            model_inputs: Feature matrix, typical format for scikit-learn (numpy array, list of lists).
            
        Returns:
            Model prediction array/matrix.
        """
        # If the model supports probabilities (e.g. classifier with predict_proba)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(model_inputs)
        return self.model.predict(model_inputs)
