"""Iterative train-only VIF feature selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditrep.preprocessing.exceptions import PreprocessingError

VIF_VERSION = "0.1.0"
VIF_INF = 1.0e12
VIF_TOLERANCE = 1.0e-9
VARIANCE_EPSILON = 1.0e-12


def _ensure_numeric_frame(X: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise PreprocessingError(f"{context} expects a pandas DataFrame, got {type(X).__name__}.")
    if X.empty or X.shape[1] == 0:
        raise PreprocessingError(f"{context} input feature matrix must not be empty.")
    if X.columns.has_duplicates:
        duplicates = sorted({str(column) for column in X.columns[X.columns.duplicated()].tolist()})
        raise PreprocessingError(f"{context} schema contains duplicate columns: {duplicates}.")
    try:
        numeric = X.astype(float)
    except ValueError as exc:
        raise PreprocessingError(f"{context} input must be fully numeric.") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise PreprocessingError(f"{context} input contains NaN or infinite values.")
    return numeric


def _single_vif(values: np.ndarray, feature_index: int) -> float:
    y = values[:, feature_index]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= VARIANCE_EPSILON:
        return VIF_INF
    others = np.delete(values, feature_index, axis=1)
    if others.shape[1] == 0:
        return 1.0
    design = np.column_stack([np.ones(values.shape[0]), others])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    ss_res = float(np.sum((y - predicted) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    if r2 >= 1.0 - VIF_TOLERANCE:
        return VIF_INF
    if r2 < 0:
        r2 = 0.0
    return float(1.0 / (1.0 - r2))


def _vif_values(frame: pd.DataFrame) -> dict[str, float]:
    values = frame.to_numpy(dtype=float)
    if frame.shape[1] == 1:
        return {str(frame.columns[0]): 1.0}
    return {str(column): _single_vif(values, index) for index, column in enumerate(frame.columns)}


class IterativeVIFSelector:
    """Remove high-VIF features iteratively using training data only."""

    def __init__(
        self,
        *,
        threshold: float = 10.0,
        minimum_features_to_keep: int = 1,
        tie_break: str = "feature_order",
    ) -> None:
        if threshold <= 1:
            raise PreprocessingError(f"VIF threshold must be > 1, got {threshold}.")
        if minimum_features_to_keep < 1:
            raise PreprocessingError("VIF minimum_features_to_keep must be >= 1.")
        if tie_break != "feature_order":
            raise PreprocessingError(f"Unsupported VIF tie_break policy {tie_break!r}.")
        self.threshold = float(threshold)
        self.minimum_features_to_keep = int(minimum_features_to_keep)
        self.tie_break = tie_break
        self._fitted = False
        self._input_feature_order: list[str] = []
        self._selected_features: list[str] = []
        self._removed_features: list[str] = []
        self._removal_history: list[dict[str, Any]] = []

    def _validate_schema(self, X: pd.DataFrame) -> None:
        columns = [str(column) for column in X.columns]
        if columns != self._input_feature_order:
            if set(columns) != set(self._input_feature_order):
                missing = sorted(set(self._input_feature_order) - set(columns))
                extra = sorted(set(columns) - set(self._input_feature_order))
                raise PreprocessingError(f"VIF transform schema mismatch; missing={missing}, extra={extra}.")
            raise PreprocessingError("VIF transform feature column order does not match fitted schema.")

    def _choose_removal(self, vif: dict[str, float], candidates: list[str]) -> str | None:
        if not vif:
            return None
        max_vif = max(vif[feature] for feature in candidates)
        if max_vif <= self.threshold:
            return None
        tied = [feature for feature in candidates if abs(vif[feature] - max_vif) <= VIF_TOLERANCE]
        return tied[0]

    def fit(self, X_train: pd.DataFrame, y_train: Any | None = None) -> "IterativeVIFSelector":
        """Fit selected features from training matrix only."""

        frame = _ensure_numeric_frame(X_train, context="VIF selector fit")
        self._input_feature_order = [str(column) for column in frame.columns]
        remaining = list(self._input_feature_order)
        self._removal_history = []
        iteration = 0

        variances = frame.var(axis=0, ddof=0)
        zero_variance_features = [
            feature for feature in remaining if float(variances[feature]) <= VARIANCE_EPSILON
        ]
        for feature in zero_variance_features:
            remaining = [candidate for candidate in remaining if candidate != feature]
            self._removal_history.append(
                {
                    "iteration": iteration,
                    "vif_values": {feature: VIF_INF},
                    "removed_feature": feature,
                    "reason": "zero_variance",
                    "remaining_features": list(remaining),
                }
            )
            iteration += 1

        if not remaining:
            raise PreprocessingError("No usable features remain after zero-variance filtering.")
        if len(remaining) < self.minimum_features_to_keep:
            raise PreprocessingError(
                "VIF minimum_features_to_keep cannot be satisfied after zero-variance filtering: "
                f"minimum_features_to_keep={self.minimum_features_to_keep}, usable_features={len(remaining)}."
            )

        while len(remaining) > self.minimum_features_to_keep:
            current = frame.loc[:, remaining]
            vif = _vif_values(current)
            removed = self._choose_removal(vif, remaining)
            reason = "vif_above_threshold" if removed is not None else None
            if removed is None:
                break
            remaining = [feature for feature in remaining if feature != removed]
            self._removal_history.append(
                {
                    "iteration": iteration,
                    "vif_values": {feature: float(value) for feature, value in vif.items()},
                    "removed_feature": removed,
                    "reason": reason,
                    "remaining_features": list(remaining),
                }
            )
            iteration += 1

        if not remaining:
            raise PreprocessingError("VIF selector removed all features.")
        self._selected_features = remaining
        self._removed_features = [feature for feature in self._input_feature_order if feature not in remaining]
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select fitted features without recomputing VIF."""

        if not self._fitted:
            raise PreprocessingError("IterativeVIFSelector.transform() called before fit().")
        frame = _ensure_numeric_frame(X, context="VIF selector transform")
        self._validate_schema(frame)
        return frame.loc[:, self._selected_features].copy(deep=True)

    def fit_transform(self, X_train: pd.DataFrame, y_train: Any | None = None) -> pd.DataFrame:
        """Fit VIF selection and transform training matrix."""

        return self.fit(X_train, y_train=y_train).transform(X_train)

    def get_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable VIF selection state."""

        return {
            "transformer": "IterativeVIFSelector",
            "version": VIF_VERSION,
            "fitted": self._fitted,
            "threshold": self.threshold,
            "minimum_features_to_keep": self.minimum_features_to_keep,
            "tie_break": self.tie_break,
            "vif_infinite_value": VIF_INF,
            "variance_epsilon": VARIANCE_EPSILON,
            "input_feature_order": list(self._input_feature_order),
            "selected_features": list(self._selected_features),
            "removed_features": list(self._removed_features),
            "removal_history": [
                {
                    "iteration": item["iteration"],
                    "vif_values": dict(item["vif_values"]),
                    "removed_feature": item["removed_feature"],
                    "reason": item["reason"],
                    "remaining_features": list(item["remaining_features"]),
                }
                for item in self._removal_history
            ],
        }
