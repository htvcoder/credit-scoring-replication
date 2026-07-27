"""Validation for P5 model configuration independent of the P2 smoke schema."""

from __future__ import annotations

from typing import Any

from creditrep.config.exceptions import ConfigError
from creditrep.config.loader import canonical_json
from creditrep.models.contract import ModelConfig
from creditrep.models.registry import MODEL_REGISTRY


def parse_model_config(payload: Any) -> ModelConfig:
    if not isinstance(payload, dict) or not payload:
        raise ConfigError("model configuration must be a non-empty mapping.")
    allowed = {"id", "random_seed", "hyperparameters", "tuning_profile"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(f"model configuration has unsupported keys: {unknown}.")
    model_id = payload.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ConfigError("model.id must be a non-empty stable model ID.")
    seed = payload.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("model.random_seed must be an integer.")
    parameters = payload.get("hyperparameters", {})
    if not isinstance(parameters, dict):
        raise ConfigError("model.hyperparameters must be a mapping.")
    try:
        canonical_json(parameters)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"model.hyperparameters must be JSON-serializable: {exc}") from exc
    profile = payload.get("tuning_profile", "reduced")
    if profile not in {"reduced", "paper_reference"}:
        raise ConfigError("model.tuning_profile must be 'reduced' or 'paper_reference'.")
    try:
        config = ModelConfig(model_id=model_id, random_seed=seed, hyperparameters=dict(parameters), tuning_profile=profile)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    capability = MODEL_REGISTRY.validate_config(config)
    for name, value in config.hyperparameters.items():
        if name in {"max_iter", "max_depth", "min_samples_leaf", "min_samples_split", "n_estimators", "n_jobs", "random_state"} and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ConfigError(f"{model_id}: hyperparameter {name} must be a positive integer.")
        if name in {"C", "learning_rate", "subsample", "colsample_bytree"} and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
            raise ConfigError(f"{model_id}: hyperparameter {name} must be a positive number.")
    if capability.supports_random_seed and "random_state" in config.hyperparameters and config.hyperparameters["random_state"] != seed:
        raise ConfigError(f"{model_id}: random_state conflicts with model.random_seed.")
    return config
