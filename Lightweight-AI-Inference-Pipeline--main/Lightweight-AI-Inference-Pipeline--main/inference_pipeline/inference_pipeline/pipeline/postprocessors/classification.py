import numpy as np
from typing import List, Dict, Any, Optional
from ..base import BasePostprocessor


class ClassificationPostprocessor(BasePostprocessor):
    """
    A postprocessor for classification tasks.
    Takes output class probabilities or logits, finds the argmax (highest probability class),
    and formats the predictions as structured dictionaries with label names and confidence scores.
    """
    def __init__(self, labels: List[str], confidence_threshold: float = 0.0):
        self.labels = labels
        self.confidence_threshold = confidence_threshold

    def postprocess(self, model_outputs: Any, metadata: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Converts probability arrays into structured classification results.
        
        Args:
            model_outputs: A numpy array or sequence of shape (batch_size, num_classes).
            metadata: Optional list of dictionaries containing request metadata (e.g. raw input string).
            
        Returns:
            A list of dictionary predictions:
            [
                {
                    "label": "positive",
                    "confidence": 0.942,
                    "probabilities": {"positive": 0.942, "negative": 0.058},
                    "threshold_passed": True
                },
                ...
            ]
        """
        outputs = np.asarray(model_outputs, dtype=np.float32)
        batch_size = outputs.shape[0]
        results = []
        
        for idx in range(batch_size):
            probs = outputs[idx]
            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            label = self.labels[max_idx] if max_idx < len(self.labels) else f"class_{max_idx}"
            
            # Map all probabilities to class names
            all_probs = {}
            for class_idx, prob in enumerate(probs):
                name = self.labels[class_idx] if class_idx < len(self.labels) else f"class_{class_idx}"
                all_probs[name] = round(float(prob), 4)

            passed = confidence >= self.confidence_threshold
            
            result = {
                "label": label if passed else "unknown",
                "confidence": round(confidence, 4),
                "probabilities": all_probs,
                "threshold_passed": passed
            }
            
            if metadata and idx < len(metadata):
                result["metadata"] = metadata[idx]
                
            results.append(result)
            
        return results
