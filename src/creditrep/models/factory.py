"""Create supported smoke classifiers."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from creditrep.models.exceptions import ModelError
from creditrep.models.registry import MODEL_REGISTRY, validate_hyperparameters


def create_model(model_type: str, parameters: dict[str, Any] | None = None, *, random_seed: int | None = None):
    capability = MODEL_REGISTRY.resolve(model_type)
    if not capability.implemented:
        raise ModelError(f"Model {model_type!r} is registered but not implemented until P5B.")
    configured = dict(parameters or {})
    validate_hyperparameters(model_type, configured)
    if random_seed is not None and "random_state" in configured and configured["random_state"] != random_seed:
        raise ModelError(f"{model_type}: random_state conflicts with requested random_seed.")
    effective = dict(capability.default_hyperparameters)
    if random_seed is not None:
        effective["random_state"] = random_seed
    effective.update(configured)
    if model_type == "logistic_regression":
        return LogisticRegression(**effective)
    if model_type == "decision_tree":
        return DecisionTreeClassifier(**effective)
    if model_type == "random_forest":
        return RandomForestClassifier(**effective)
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ModelError("xgboost is required for model.type='xgboost'.") from exc
        return XGBClassifier(**effective)
    if model_type in {"mlp_1", "mlp_3", "mlp_5"}:
        from creditrep.models.neural.wrapper import MLPProbabilityEstimator
        mapped = dict(effective)
        if random_seed is not None:
            mapped["random_seed"] = random_seed
        mapped.pop("random_state", None)
        return MLPProbabilityEstimator(model_type, **mapped)
    raise ModelError(f"Unsupported model type: {model_type}")
