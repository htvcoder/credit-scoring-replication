"""Typed, JSON-safe contracts shared by Phase 5 classical models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from creditrep.config.loader import sha256_canonical
from creditrep.models.exceptions import ModelError


ModelId = Literal["logistic_regression", "decision_tree", "random_forest", "xgboost"]
TuningProfile = Literal["reduced", "paper_reference"]


@runtime_checkable
class ProbabilityEstimator(Protocol):
    """Minimum sklearn-compatible estimator interface required by P5."""

    classes_: Any

    def fit(self, X: Any, y: Any) -> "ProbabilityEstimator": ...

    def predict(self, X: Any) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...


@dataclass(frozen=True)
class ModelCapability:
    model_id: ModelId
    display_name: str
    family: str
    estimator_name: str
    library_name: str
    supports_random_seed: bool
    supports_probability: bool
    expected_classes: tuple[int, int] = (0, 1)
    allowed_hyperparameters: tuple[str, ...] = ()
    default_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    algorithm: str | None = None
    implementation: str | None = None
    replication_role: str = "replication_model"
    deviation_from_paper: str | None = None
    implemented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    model_id: ModelId
    random_seed: int
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    tuning_profile: TuningProfile = "reduced"

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "hyperparameters": dict(self.hyperparameters),
            "model_id": self.model_id,
            "random_seed": self.random_seed,
            "tuning_profile": self.tuning_profile,
        }

    @property
    def config_hash(self) -> str:
        return sha256_canonical(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return self.canonical_payload() | {"config_hash": self.config_hash}


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Normalized metadata for model-validation artifacts; always JSON-safe."""

    model_id: str
    model_family: str
    estimator_name: str
    library_name: str
    library_version: str | None
    configured_hyperparameters: Mapping[str, Any]
    effective_hyperparameters: Mapping[str, Any]
    random_seed: int
    expected_classes: tuple[int, int]
    observed_classes: tuple[int, ...] | None
    algorithm: str | None = None
    implementation: str | None = None
    replication_role: str = "replication_model"
    deviation_from_paper: str | None = None
    tuning_profile: str = "reduced"
    positive_class: int = 1
    probability_mapping: str = "P(class 1) = P(bad/default)"
    selected_hyperparameters: Mapping[str, Any] | None = None
    tuning_metadata: Mapping[str, Any] | None = None
    fit_duration_seconds: float | None = None
    prediction_duration_seconds: float | None = None
    convergence_status: str | None = None
    warnings: tuple[str, ...] = ()
    result_scope: str = "model_validation"
    publishable: bool = False
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe(asdict(self))
        # Validate early so artifact writers never receive non-JSON-safe values.
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        return payload


def _json_safe(value: Any) -> Any:
    """Normalize library metadata (including NumPy scalars/NaN) for JSON artifacts."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def positive_class_probabilities(model: Any, probabilities: Any, *, expected_rows: int) -> np.ndarray:
    """Return P(class 1), mapping by ``classes_`` rather than column position."""

    if not hasattr(model, "classes_"):
        raise ModelError("Probability estimator must expose classes_ after fit.")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ModelError(f"Probability estimator classes_ must include class 1, got {classes}.")
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != len(classes):
        raise ModelError(
            f"predict_proba returned shape {values.shape}; expected (n_rows, {len(classes)}) for classes_."
        )
    if values.shape[0] != expected_rows:
        raise ModelError(f"Probability row count mismatch: {values.shape[0]} != {expected_rows}.")
    if not np.isfinite(values).all():
        raise ModelError("predict_proba returned NaN or Infinity.")
    if ((values < 0) | (values > 1)).any():
        raise ModelError("predict_proba values must be in [0, 1].")
    return values[:, classes.index(1)].astype(float)
