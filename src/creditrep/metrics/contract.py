"""Typed JSON-serializable metric result contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MetricStatus(str, Enum):
    VALID = "valid"
    UNDEFINED = "undefined"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class MetricExactness(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    NOT_APPLICABLE = "not_applicable"


def _enum_value(value: Enum | str, enum_type: type[Enum], *, field_name: str) -> str:
    if isinstance(value, enum_type):
        return str(value.value)
    if isinstance(value, str):
        allowed = {item.value for item in enum_type}
        if value in allowed:
            return value
    allowed = sorted(item.value for item in enum_type)
    raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}.")


def _json_compatible(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or Infinity.")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item, field_name=field_name) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings.")
            normalized[key] = _json_compatible(item, field_name=field_name)
        return normalized
    raise ValueError(f"{field_name} must be JSON-compatible, got {type(value).__name__}.")


def _finite_optional_float(value: float | None, *, status: str) -> float | None:
    if value is None:
        if status == MetricStatus.VALID.value:
            raise ValueError("MetricResult.value is required when status is valid.")
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("MetricResult.value must be finite.")
    return numeric


@dataclass(frozen=True)
class MetricSpecification:
    metric_id: str
    metric_version: str
    direction: MetricDirection | str
    exactness: MetricExactness | str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric_id.strip():
            raise ValueError("metric_id is required.")
        if not self.metric_version.strip():
            raise ValueError("metric_version is required.")
        object.__setattr__(self, "direction", _enum_value(self.direction, MetricDirection, field_name="direction"))
        object.__setattr__(self, "exactness", _enum_value(self.exactness, MetricExactness, field_name="exactness"))
        object.__setattr__(self, "parameters", _json_compatible(dict(self.parameters), field_name="parameters"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "exactness": self.exactness,
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    metric_version: str
    value: float | None
    direction: MetricDirection | str
    status: MetricStatus | str
    parameters: dict[str, Any] = field(default_factory=dict)
    exactness: MetricExactness | str = MetricExactness.NOT_APPLICABLE
    warnings: tuple[str, ...] | list[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.metric_id.strip():
            raise ValueError("metric_id is required.")
        if not self.metric_version.strip():
            raise ValueError("metric_version is required.")
        status = _enum_value(self.status, MetricStatus, field_name="status")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "direction", _enum_value(self.direction, MetricDirection, field_name="direction"))
        object.__setattr__(self, "exactness", _enum_value(self.exactness, MetricExactness, field_name="exactness"))
        object.__setattr__(self, "value", _finite_optional_float(self.value, status=status))
        object.__setattr__(self, "parameters", _json_compatible(dict(self.parameters), field_name="parameters"))
        normalized_warnings = tuple(str(item) for item in self.warnings)
        object.__setattr__(self, "warnings", normalized_warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "exactness": self.exactness,
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "parameters": self.parameters,
            "status": self.status,
            "value": self.value,
            "warnings": list(self.warnings),
        }
