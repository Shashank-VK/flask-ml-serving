"""Model loader and inference service."""

import json
import os
import pickle
from typing import Any, Dict, List
import numpy as np


class ModelService:
    """Manages model artifact lifecycle and inference computations."""

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "model.pkl")
        self.metrics_path = os.path.join(artifacts_dir, "metrics.json")
        self.model = None
        self.metrics: Dict[str, Any] = {}
        self.features: List[str] = []
        self.classes: List[str] = []
        self.load()

    def load(self) -> None:
        """Load model and metrics from disk."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        if not os.path.isfile(self.metrics_path):
            raise FileNotFoundError(f"Metrics artifact not found at {self.metrics_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.metrics_path, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

        self.features = list(self.metrics.get("features", []))
        self.classes = list(self.metrics.get("classes", []))

    def predict(self, feature_values: List[float]) -> Dict[str, Any]:
        """Perform inference on the provided feature vector."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        X = np.array([feature_values], dtype=float)
        class_idx = int(self.model.predict(X)[0])
        pred_label = self.classes[class_idx] if class_idx < len(self.classes) else f"class_{class_idx}"

        probabilities_raw = self.model.predict_proba(X)[0]
        probabilities = {
            self.classes[i]: float(probabilities_raw[i])
            for i in range(len(self.classes))
        }

        return {
            "prediction": pred_label,
            "class_index": class_idx,
            "probabilities": probabilities,
        }
