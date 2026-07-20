"""Minimal P2C smoke metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from creditrep.evaluation.exceptions import EvaluationError


def _finite_float(value: float, *, name: str) -> float:
    if not math.isfinite(float(value)):
        raise EvaluationError(f"Metric {name} is not finite: {value!r}.")
    return float(value)


def compute_binary_metrics(
    *,
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    if set(np.unique(y_true_array)) != {0, 1}:
        raise EvaluationError(f"Metrics require y_true classes {{0, 1}}, got {sorted(np.unique(y_true_array))}.")
    metrics = {
        "roc_auc": _finite_float(roc_auc_score(y_true_array, y_score), name="roc_auc"),
        "accuracy": _finite_float(accuracy_score(y_true_array, y_pred), name="accuracy"),
        "precision": _finite_float(
            precision_score(y_true_array, y_pred, zero_division=0),
            name="precision",
        ),
        "recall": _finite_float(recall_score(y_true_array, y_pred, zero_division=0), name="recall"),
        "f1": _finite_float(f1_score(y_true_array, y_pred, zero_division=0), name="f1"),
        "log_loss": _finite_float(log_loss(y_true_array, y_score, labels=[0, 1]), name="log_loss"),
        "brier_score": _finite_float(brier_score_loss(y_true_array, y_score), name="brier_score"),
        "test_row_count": int(len(y_true_array)),
        "positive_class_count": int((y_true_array == 1).sum()),
        "negative_class_count": int((y_true_array == 0).sum()),
        "classification_threshold": float(threshold),
        "predicted_positive_rate": _finite_float(float((y_pred == 1).sum() / len(y_pred)), name="predicted_positive_rate"),
    }
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }
    return metrics
