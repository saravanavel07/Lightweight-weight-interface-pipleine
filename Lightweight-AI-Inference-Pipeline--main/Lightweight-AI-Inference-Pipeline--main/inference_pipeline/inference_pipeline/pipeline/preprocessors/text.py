import re
import numpy as np
from typing import List, Dict, Union, Any, Optional
from ..base import BasePreprocessor


class SimpleTextPreprocessor(BasePreprocessor):
    """
    A zero-dependency, pure Python/NumPy TF-IDF Text Vectorizer.
    Converts list of raw strings to a dense numerical feature matrix.
    Loads vocabulary mapping and IDF weights.
    """
    def __init__(self, vocab: Dict[str, int], idf: List[float]):
        self.vocab = vocab
        self.idf = np.array(idf, dtype=np.float32)
        self.vocab_size = len(vocab)
        
    @classmethod
    def from_config(cls, config_path: str) -> "SimpleTextPreprocessor":
        """Loads vocabulary and idf weights from a JSON configuration."""
        import json
        with open(config_path, "r") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"], idf=data["idf"])

    def _clean_and_tokenize(self, text: str) -> List[str]:
        """Cleans text, lowcases it, and splits it into tokens."""
        text = str(text).lower()
        # Find all word characters
        words = re.findall(r'\b\w+\b', text)
        return words

    def preprocess(self, inputs: List[str]) -> np.ndarray:
        """
        Transforms a batch of texts into a TF-IDF matrix.
        
        Args:
            inputs: A list of text strings.
            
        Returns:
            A NumPy array of shape (batch_size, vocab_size).
        """
        batch_size = len(inputs)
        features = np.zeros((batch_size, self.vocab_size), dtype=np.float32)
        
        for idx, text in enumerate(inputs):
            tokens = self._clean_and_tokenize(text)
            if not tokens:
                continue
            
            # Count term frequencies (TF) in this document
            tf = {}
            for token in tokens:
                if token in self.vocab:
                    tf[token] = tf.get(token, 0) + 1
            
            # Write TF values to features row
            for token, count in tf.items():
                vocab_idx = self.vocab[token]
                # Log normalization for term frequency: 1 + log(tf)
                features[idx, vocab_idx] = 1 + np.log(count)

        # Apply inverse document frequency (IDF) weighting: TF * IDF
        features = features * self.idf
        
        # L2 normalize each document vector (row-wise normalization)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        features = features / norms
        
        return features
