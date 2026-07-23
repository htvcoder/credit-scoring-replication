"""Common validation helpers for binary probability metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class MetricInputError(ValueError):
    """Raised when metric inputs violate the Phase 4 contract."""


@dataclass(frozen=True)
class ValidatedBinaryProbabilityInputs:
    y_true: np.ndarray
    y_score: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.y_true.size)

    @property
    def positive_count(self) -> int:
        return int(np.sum(self.y_true == 1))

    @property
    def negative_count(self) -> int:
        return int(np.sum(self.y_true == 0))


def _as_1d_numeric_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise MetricInputError(f"{name} must be one-dimensional, got ndim={array.ndim}.")
    if array.size == 0:
        raise MetricInputError(f"{name} must not be empty.")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise MetricInputError(f"{name} must be numeric.") from exc
    if not np.isfinite(numeric).all():
        raise MetricInputError(f"{name} must not contain NaN or Infinity.")
    return numeric


def validate_binary_probability_inputs(y_true: Any, y_score: Any) -> ValidatedBinaryProbabilityInputs:
    true_array = _as_1d_numeric_array(y_true, name="y_true")
    score_array = _as_1d_numeric_array(y_score, name="y_score")
    if true_array.shape[0] != score_array.shape[0]:
        raise MetricInputError(
            f"y_true and y_score must have the same length, got {true_array.shape[0]} and {score_array.shape[0]}."
        )
    unique_labels = set(np.unique(true_array).tolist())
    if unique_labels != {0.0, 1.0} and unique_labels not in ({0.0}, {1.0}):
        normalized = sorted(int(value) if float(value).is_integer() else value for value in unique_labels)
        raise MetricInputError(f"y_true must contain only binary labels {{0, 1}}, got {normalized}.")
    if ((score_array < 0.0) | (score_array > 1.0)).any():
        raise MetricInputError("y_score must contain probabilities in [0, 1] without clipping.")
    return ValidatedBinaryProbabilityInputs(y_true=true_array.astype(np.int64), y_score=score_array.astype(float))


def validate_partial_gini_cutoff(b: Any) -> float:
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        raise MetricInputError(f"b must be numeric, got {type(b).__name__}.")
    cutoff = float(b)
    if not math.isfinite(cutoff):
        raise MetricInputError("b must be finite.")
    if cutoff <= 0.0 or cutoff > 1.0:
        raise MetricInputError(f"b must satisfy 0 < b <= 1, got {cutoff}.")
    return cutoff
