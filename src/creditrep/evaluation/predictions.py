"""Prediction validation, serialization payloads and hashes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditrep.config.loader import sha256_canonical
from creditrep.evaluation.exceptions import EvaluationError


def validate_positive_class_probabilities(model: Any, probabilities: np.ndarray, *, expected_rows: int) -> np.ndarray:
    if not hasattr(model, "classes_"):
        raise EvaluationError("Model must expose classes_ after fit.")
    classes = list(model.classes_)
    if 1 not in classes:
        raise EvaluationError(f"Model classes_ must include positive class 1, got {classes}.")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise EvaluationError(
            f"predict_proba returned shape {probabilities.shape}, expected two-dimensional probabilities for classes_."
        )
    if probabilities.shape[0] != expected_rows:
        raise EvaluationError(f"Prediction length mismatch: {probabilities.shape[0]} != {expected_rows}.")
    if not np.isfinite(probabilities).all():
        raise EvaluationError("Predicted probabilities contain NaN or Infinity.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise EvaluationError("Predicted probabilities must be in [0, 1].")
    return probabilities[:, classes.index(1)].astype(float)


def build_prediction_frame(
    *,
    row_positions: tuple[int, ...],
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred = (y_score >= threshold).astype(int)
    frame = pd.DataFrame(
        {
            "row_position": list(row_positions),
            "partition": ["test"] * len(row_positions),
            "y_true": y_true_array.tolist(),
            "y_score": y_score.tolist(),
            "y_pred": y_pred.tolist(),
        }
    )
    validate_prediction_frame(frame, expected_rows=len(row_positions))
    return frame


def validate_prediction_frame(frame: pd.DataFrame, *, expected_rows: int | None = None) -> None:
    expected_columns = ["row_position", "partition", "y_true", "y_score", "y_pred"]
    if list(frame.columns) != expected_columns:
        raise EvaluationError(f"Predictions must have columns {expected_columns}.")
    if expected_rows is not None and len(frame) != expected_rows:
        raise EvaluationError(f"Prediction row count mismatch: {len(frame)} != {expected_rows}.")
    if set(frame["partition"].unique()) != {"test"}:
        raise EvaluationError("Prediction partition must always be 'test'.")
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise EvaluationError("y_true must be binary.")
    if not set(frame["y_pred"].unique()).issubset({0, 1}):
        raise EvaluationError("y_pred must be binary.")
    scores = frame["y_score"].astype(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise EvaluationError("y_score must be finite and in [0, 1].")
    if frame["row_position"].duplicated().any():
        raise EvaluationError("Prediction row_position values must be unique.")


def prediction_hash(frame: pd.DataFrame, *, split_hash: str, model_config_hash: str) -> str:
    payload = {
        "model_config_hash": model_config_hash,
        "predictions": [
            {
                "row_position": int(row["row_position"]),
                "y_pred": int(row["y_pred"]),
                "y_score": round(float(row["y_score"]), 15),
                "y_true": int(row["y_true"]),
            }
            for _, row in frame.sort_values("row_position").iterrows()
        ],
        "split_hash": split_hash,
    }
    return sha256_canonical(payload)
