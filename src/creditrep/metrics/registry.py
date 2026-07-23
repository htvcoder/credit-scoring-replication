"""Metric registry and deterministic config contract for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from creditrep.config.exceptions import ConfigError
from creditrep.config.loader import sha256_canonical
from creditrep.metrics.calibration import BRIER_SCORE_SPEC, compute_brier_score
from creditrep.metrics.contract import MetricResult, MetricSpecification
from creditrep.metrics.discrimination import PARTIAL_GINI_SPEC, ROC_AUC_SPEC, compute_partial_gini, compute_roc_auc, partial_gini_specification
from creditrep.metrics.profit import EMP_SPEC, compute_emp


@dataclass(frozen=True)
class MetricConfig:
    metric_id: str
    parameters: dict[str, Any]

    def canonical_payload(self) -> dict[str, Any]:
        return {"id": self.metric_id, "parameters": dict(sorted(self.parameters.items()))}


_ALLOWED_PARAMETERS = {
    "roc_auc": frozenset(),
    "brier_score": frozenset(),
    "partial_gini": frozenset({"b"}),
    "emp": frozenset(),
}


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping.")
    return value


def _resolve_metric_parameters(metric_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if metric_id == "partial_gini":
        if "b" not in parameters:
            return {"b": 0.4}
        b = parameters["b"]
        if not isinstance(b, (int, float)) or isinstance(b, bool):
            raise ConfigError("metrics[].parameters.b must be numeric.")
        b = float(b)
        if not 0.0 < b < 1.0:
            raise ConfigError("metrics[].parameters.b must be > 0 and < 1.")
        return {"b": b}
    return {}


def parse_metric_configs(payload: Any) -> tuple[MetricConfig, ...]:
    if not isinstance(payload, list) or not payload:
        raise ConfigError("metrics must be a non-empty list.")
    configs: list[MetricConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        mapping = _mapping(item, context=f"metrics[{index}]")
        unknown = sorted(set(mapping) - {"id", "parameters"})
        if unknown:
            raise ConfigError(f"metrics[{index}] has unsupported keys: {unknown}.")
        metric_id = mapping.get("id")
        if not isinstance(metric_id, str) or not metric_id.strip():
            raise ConfigError(f"metrics[{index}].id is required.")
        metric_id = metric_id.strip()
        if metric_id not in _ALLOWED_PARAMETERS:
            raise ConfigError(f"Unsupported metric id {metric_id!r}.")
        if metric_id in seen:
            raise ConfigError(f"Duplicate metric id {metric_id!r} is not allowed.")
        seen.add(metric_id)
        raw_parameters = mapping.get("parameters", {})
        if raw_parameters is None:
            raw_parameters = {}
        parameters = _mapping(raw_parameters, context=f"metrics[{index}].parameters")
        unknown_parameters = sorted(set(parameters) - _ALLOWED_PARAMETERS[metric_id])
        if unknown_parameters:
            raise ConfigError(f"{metric_id}: unsupported parameters: {unknown_parameters}.")
        configs.append(MetricConfig(metric_id=metric_id, parameters=_resolve_metric_parameters(metric_id, parameters)))
    return tuple(configs)


def metric_config_hash(metrics: tuple[MetricConfig, ...]) -> str:
    return sha256_canonical([item.canonical_payload() for item in metrics])


def get_metric_specification(metric_id: str, *, parameters: dict[str, Any] | None = None) -> MetricSpecification:
    resolved = dict(parameters or {})
    if metric_id == "roc_auc":
        return ROC_AUC_SPEC
    if metric_id == "brier_score":
        return BRIER_SCORE_SPEC
    if metric_id == "partial_gini":
        return partial_gini_specification(b=resolved.get("b", 0.4))
    if metric_id == "emp":
        return EMP_SPEC
    raise ConfigError(f"Unsupported metric id {metric_id!r}.")


def compute_configured_metric(metric_config: MetricConfig, y_true: Any, y_score: Any) -> MetricResult:
    if metric_config.metric_id == "roc_auc":
        return compute_roc_auc(y_true, y_score)
    if metric_config.metric_id == "brier_score":
        return compute_brier_score(y_true, y_score)
    if metric_config.metric_id == "partial_gini":
        return compute_partial_gini(y_true, y_score, b=float(metric_config.parameters["b"]))
    if metric_config.metric_id == "emp":
        return compute_emp(y_true, y_score, parameters=metric_config.parameters)
    raise ConfigError(f"Unsupported metric id {metric_config.metric_id!r}.")
