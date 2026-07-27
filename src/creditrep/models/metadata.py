"""Deterministic, JSON-safe metadata extraction for fitted P5 estimators."""

from __future__ import annotations

from typing import Any

import sklearn

from creditrep.models.contract import ModelArtifactMetadata
from creditrep.models.registry import MODEL_REGISTRY


def build_model_metadata(model: Any, *, model_id: str, configured_hyperparameters: dict[str, Any], random_seed: int, fit_duration_seconds: float | None = None, prediction_duration_seconds: float | None = None, warnings: tuple[str, ...] = (), tuning_profile: str = "reduced", result_scope: str = "model_validation") -> dict[str, Any]:
    capability = MODEL_REGISTRY.resolve(model_id)
    version = sklearn.__version__
    if capability.library_name == "xgboost":
        import xgboost
        version = xgboost.__version__
    effective = model.get_params(deep=False) if hasattr(model, "get_params") else configured_hyperparameters
    observed = tuple(int(item) for item in model.classes_) if hasattr(model, "classes_") else None
    convergence_status = "unknown"
    if model_id != "logistic_regression":
        convergence_status = "not_applicable"
    return ModelArtifactMetadata(model_id=model_id, model_family=capability.family, estimator_name=capability.estimator_name, library_name=capability.library_name, library_version=version, configured_hyperparameters=configured_hyperparameters, effective_hyperparameters=effective, random_seed=random_seed, expected_classes=capability.expected_classes, observed_classes=observed, algorithm=capability.algorithm, implementation=capability.implementation, replication_role=capability.replication_role, deviation_from_paper=capability.deviation_from_paper, tuning_profile=tuning_profile, fit_duration_seconds=fit_duration_seconds, prediction_duration_seconds=prediction_duration_seconds, convergence_status=convergence_status, warnings=warnings, result_scope=result_scope, publishable=False).to_dict()
