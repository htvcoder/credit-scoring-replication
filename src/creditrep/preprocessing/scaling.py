"""Optional train-only standard scaling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditrep.preprocessing.exceptions import PreprocessingError
from creditrep.preprocessing.vif import _ensure_numeric_frame

SCALER_VERSION = "0.1.0"


class TrainOnlyStandardScaler:
    """Standardize features using means and scales learned from training data."""

    def __init__(self, *, enabled: bool = True, strategy: str = "standard") -> None:
        if strategy != "standard":
            raise PreprocessingError(f"Unsupported scaling strategy {strategy!r}.")
        self.enabled = bool(enabled)
        self.strategy = strategy
        self._fitted = False
        self._feature_order: list[str] = []
        self._means: dict[str, float] = {}
        self._scales: dict[str, float] = {}
        self._zero_variance_features: list[str] = []

    def _validate_schema(self, X: pd.DataFrame) -> None:
        columns = [str(column) for column in X.columns]
        if columns != self._feature_order:
            if set(columns) != set(self._feature_order):
                missing = sorted(set(self._feature_order) - set(columns))
                extra = sorted(set(columns) - set(self._feature_order))
                raise PreprocessingError(f"Scaler transform schema mismatch; missing={missing}, extra={extra}.")
            raise PreprocessingError("Scaler transform feature column order does not match fitted schema.")

    def fit(self, X_train: pd.DataFrame, y_train: Any | None = None) -> "TrainOnlyStandardScaler":
        """Fit means and scales from training matrix only."""

        frame = _ensure_numeric_frame(X_train, context="Scaler fit")
        self._feature_order = [str(column) for column in frame.columns]
        self._means = {}
        self._scales = {}
        self._zero_variance_features = []
        for column in self._feature_order:
            mean = float(frame[column].mean())
            scale = float(frame[column].std(ddof=0))
            if not np.isfinite(mean) or not np.isfinite(scale):
                raise PreprocessingError(f"Scaler feature {column!r} produced non-finite mean/scale.")
            if scale == 0:
                self._zero_variance_features.append(column)
                scale = 1.0
            self._means[column] = mean
            self._scales[column] = scale
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale with fitted training statistics, or passthrough when disabled."""

        if not self._fitted:
            raise PreprocessingError("TrainOnlyStandardScaler.transform() called before fit().")
        frame = _ensure_numeric_frame(X, context="Scaler transform")
        self._validate_schema(frame)
        if not self.enabled:
            return frame.loc[:, self._feature_order].copy(deep=True)
        output = frame.copy(deep=True)
        for column in self._feature_order:
            output[column] = (output[column] - self._means[column]) / self._scales[column]
        if not np.isfinite(output.to_numpy()).all():
            raise PreprocessingError("Scaler output contains NaN or infinite values.")
        return output.loc[:, self._feature_order]

    def fit_transform(self, X_train: pd.DataFrame, y_train: Any | None = None) -> pd.DataFrame:
        """Fit scaler and transform training matrix."""

        return self.fit(X_train, y_train=y_train).transform(X_train)

    def get_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable scaler state."""

        return {
            "transformer": "TrainOnlyStandardScaler",
            "version": SCALER_VERSION,
            "fitted": self._fitted,
            "enabled": self.enabled,
            "strategy": self.strategy,
            "feature_order": list(self._feature_order),
            "training_means": dict(self._means),
            "training_scales": dict(self._scales),
            "zero_variance_handling": "scale set to 1.0",
            "zero_variance_features": list(self._zero_variance_features),
        }
