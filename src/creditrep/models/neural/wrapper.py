"""Shared sklearn-shaped adapter for P6B depth specifications."""

from __future__ import annotations
from typing import Any
import numpy as np
from creditrep.models.neural.exceptions import NeuralInputError
from creditrep.models.neural.specifications import get_mlp_specification
from creditrep.models.neural.training import train_mlp


class MLPProbabilityEstimator:
    classes_ = np.array([0, 1])

    def __init__(self, model_id: str, **parameters: Any):
        self.spec = get_mlp_specification(model_id)
        self.config = self.spec.config(**parameters)
        self._result = None

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_validation: Any | None = None,
        y_validation: Any | None = None,
    ):
        self._result = train_mlp(
            X, y, self.config, X_validation=X_validation, y_validation=y_validation
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._result is None:
            raise NeuralInputError("MLP estimator must be fitted before predict_proba.")
        p = self._result.predict_probabilities(X)
        return np.column_stack((1 - p, p))

    def get_training_summary(self):
        if self._result is None:
            raise NeuralInputError("MLP estimator must be fitted.")
        return self._result.summary.to_dict()

    def get_training_history(self):
        if self._result is None:
            raise NeuralInputError("MLP estimator must be fitted.")
        return self._result.history.to_dict()

    def get_model_metadata(self):
        data = self.spec.to_dict()
        data["config"] = self.config.to_dict()
        if self._result is not None:
            data["training_summary"] = self.get_training_summary()
        return data
