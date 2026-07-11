import json
import numpy as np
from typing import Any
from ..base import BaseBackend

class NumpyMLPBackend(BaseBackend):
    """
    A zero-dependency, pure NumPy forward-pass implementation of a Multi-Layer Perceptron (MLP).
    Loads weights and biases from a JSON file, facilitating lightweight deep-learning-like inference.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.w1 = np.array([])
        self.b1 = np.array([])
        self.w2 = np.array([])
        self.b2 = np.array([])
        self.load_model()

    def load_model(self):
        """Loads weights and biases from the configured JSON file."""
        with open(self.model_path, 'r') as f:
            model_data = json.load(f)
        
        self.w1 = np.array(model_data["W1"])
        self.b1 = np.array(model_data["b1"])
        self.w2 = np.array(model_data["W2"])
        self.b2 = np.array(model_data["b2"])

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability (avoid overflow)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(self, model_inputs: Any) -> np.ndarray:
        """
        Executes a forward pass on the input features.
        
        Args:
            model_inputs: A NumPy array of shape (batch_size, input_dim)
            
        Returns:
            A NumPy array of shape (batch_size, num_classes) representing class probabilities.
        """
        X = np.asarray(model_inputs, dtype=np.float32)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0) # Reshape 1D vector to 2D batch of 1
            
        # Hidden layer 1: Relu(X * W1 + b1)
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer: Softmax(a1 * W2 + b2)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.softmax(z2)
        
        return a2
