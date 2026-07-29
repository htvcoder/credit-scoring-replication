"""Typed configuration for non-publishable P5C model validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from creditrep.config.exceptions import ConfigError
from creditrep.config.loader import sha256_canonical
from creditrep.metrics.registry import (
    MetricConfig,
    get_metric_specification,
    parse_metric_configs,
)
from creditrep.models.registry import MODEL_REGISTRY, validate_hyperparameters


@dataclass(frozen=True)
class ModelValidationConfig:
    experiment_name: str
    dataset_id: str
    output_root: str
    model_candidates: dict[str, tuple[dict[str, Any], ...]]
    metrics: tuple[MetricConfig, ...]
    optimization_metric: str
    threshold: float
    random_seed: int
    protocol_config_path: str = "configs/protocols/protocol_a.yaml"
    outer_n_repeats: int = 1
    outer_n_splits: int = 2
    inner_n_splits: int = 2
    result_scope: str = "model_validation"
    publishable: bool = False
    resume: bool = False
    max_retry_attempts: int = 1

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "experiment": {
                "name": self.experiment_name,
                "result_scope": self.result_scope,
                "publishable": self.publishable,
            },
            "dataset": {"id": self.dataset_id},
            "output": {"root_dir": self.output_root},
            "preprocessing": {"protocol_config": self.protocol_config_path},
            "cross_validation": {
                "outer": {
                    "strategy": "repeated_stratified_2fold",
                    "n_repeats": self.outer_n_repeats,
                    "n_splits": self.outer_n_splits,
                    "shuffle": True,
                    "random_seed": self.random_seed,
                },
                "inner": {
                    "strategy": "stratified_kfold",
                    "n_splits": self.inner_n_splits,
                    "shuffle": True,
                    "random_seed_policy": "derived_from_outer",
                },
            },
            "models": {
                key: list(value) for key, value in sorted(self.model_candidates.items())
            },
            "metrics": [item.canonical_payload() for item in self.metrics],
            "optimization_metric": self.optimization_metric,
            "threshold": self.threshold,
            "random_seed": self.random_seed,
            "resume": self.resume,
            "retry_policy": {"max_retry_attempts": self.max_retry_attempts},
        }

    @property
    def config_hash(self) -> str:
        return sha256_canonical(self.canonical_payload())


def parse_model_validation_config(payload: Any) -> ModelValidationConfig:
    if not isinstance(payload, dict):
        raise ConfigError("model validation config must be a mapping.")
    allowed = {
        "experiment",
        "dataset",
        "output",
        "preprocessing",
        "cross_validation",
        "models",
        "metrics",
        "optimization_metric",
        "threshold",
        "random_seed",
        "resume",
        "retry_policy",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(f"model validation config has unsupported keys: {unknown}.")
    experiment, dataset, output, models = (
        payload.get(key) for key in ("experiment", "dataset", "output", "models")
    )
    if not all(
        isinstance(value, dict) for value in (experiment, dataset, output, models)
    ):
        raise ConfigError("experiment, dataset, output and models must be mappings.")
    if (
        experiment.get("publishable") is not False
        or experiment.get("result_scope") != "model_validation"
    ):
        raise ConfigError(
            "P5C requires publishable=false and result_scope=model_validation."
        )
    name, dataset_id, output_root = (
        experiment.get("name"),
        dataset.get("id"),
        output.get("root_dir"),
    )
    if not all(
        isinstance(value, str) and value.strip()
        for value in (name, dataset_id, output_root)
    ):
        raise ConfigError(
            "experiment.name, dataset.id and output.root_dir are required."
        )
    candidates: dict[str, tuple[dict[str, Any], ...]] = {}
    for model_id, values in models.items():
        MODEL_REGISTRY.resolve(model_id)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"models.{model_id} must be a non-empty candidate list.")
        if model_id in candidates:
            raise ConfigError(f"Duplicate model ID: {model_id}.")
        parsed = []
        for value in values:
            if not isinstance(value, dict):
                raise ConfigError(f"models.{model_id} candidates must be mappings.")
            validate_hyperparameters(model_id, value)
            parsed.append(dict(value))
        candidates[model_id] = tuple(parsed)
    metrics = parse_metric_configs(payload.get("metrics"))
    optimization_metric = payload.get("optimization_metric")
    if (
        optimization_metric not in {item.metric_id for item in metrics}
        or optimization_metric == "emp"
    ):
        raise ConfigError(
            "optimization_metric must be a configured supported non-EMP metric."
        )
    get_metric_specification(
        optimization_metric,
        parameters=next(
            item.parameters for item in metrics if item.metric_id == optimization_metric
        ),
    )
    threshold = payload.get("threshold", 0.5)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 <= threshold <= 1
    ):
        raise ConfigError("threshold must be in [0, 1].")
    seed = payload.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("random_seed must be an integer.")
    resume = payload.get("resume", False)
    if not isinstance(resume, bool):
        raise ConfigError("resume must be true or false.")
    retry_policy = payload.get("retry_policy", {})
    if not isinstance(retry_policy, dict):
        raise ConfigError("retry_policy must be a mapping.")
    max_retry_attempts = retry_policy.get("max_retry_attempts", 1)
    if (
        not isinstance(max_retry_attempts, int)
        or isinstance(max_retry_attempts, bool)
        or max_retry_attempts < 0
    ):
        raise ConfigError(
            "retry_policy.max_retry_attempts must be a non-negative integer."
        )
    preprocessing = payload.get("preprocessing", {})
    cv = payload.get("cross_validation", {})
    if not isinstance(preprocessing, dict) or not isinstance(cv, dict):
        raise ConfigError("preprocessing and cross_validation must be mappings.")
    protocol = preprocessing.get("protocol_config", "configs/protocols/protocol_a.yaml")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ConfigError("preprocessing.protocol_config must be a non-empty path.")
    outer, inner = cv.get("outer", {}), cv.get("inner", {})
    if not isinstance(outer, dict) or not isinstance(inner, dict):
        raise ConfigError(
            "cross_validation.outer and cross_validation.inner must be mappings."
        )
    outer_repeats, outer_splits, inner_splits = (
        outer.get("n_repeats", 1),
        outer.get("n_splits", 2),
        inner.get("n_splits", 2),
    )
    if (
        outer.get("strategy", "repeated_stratified_2fold")
        != "repeated_stratified_2fold"
        or outer_splits != 2
        or not isinstance(outer_repeats, int)
        or outer_repeats < 1
        or inner.get("strategy", "stratified_kfold") != "stratified_kfold"
        or not isinstance(inner_splits, int)
        or inner_splits < 2
    ):
        raise ConfigError(
            "P5C requires repeated_stratified_2fold outer CV and inner stratified_kfold with at least 2 splits."
        )
    return ModelValidationConfig(
        name.strip(),
        dataset_id.strip().upper(),
        output_root.strip(),
        candidates,
        metrics,
        optimization_metric,
        float(threshold),
        seed,
        protocol.strip(),
        outer_repeats,
        outer_splits,
        inner_splits,
        resume=resume,
        max_retry_attempts=max_retry_attempts,
    )
