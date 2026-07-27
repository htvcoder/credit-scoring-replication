"""Create supported smoke classifiers."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from creditrep.models.exceptions import ModelError
from creditrep.models.registry import MODEL_REGISTRY


def create_model(model_type: str, parameters: dict[str, Any]):
    capability = MODEL_REGISTRY.resolve(model_type)
    if not capability.implemented:
        raise ModelError(f"Model {model_type!r} is registered but not implemented until P5B.")
    if model_type == "logistic_regression":
        return LogisticRegression(**parameters)
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ModelError("xgboost is required for model.type='xgboost'.") from exc
        return XGBClassifier(**parameters)
    raise ModelError(f"Unsupported model type: {model_type}")
