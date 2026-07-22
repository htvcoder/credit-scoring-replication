"""Train-only Weight of Evidence encoding for categorical features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditrep.preprocessing.exceptions import PreprocessingError

WOE_VERSION = "0.1.0"
WOE_SIGN_CONVENTION = "good_over_bad"


def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise PreprocessingError(f"WOE encoder expects a pandas DataFrame, got {type(X).__name__}.")
    if X.empty:
        raise PreprocessingError("WOE encoder input feature matrix must not be empty.")
    if X.columns.has_duplicates:
        duplicates = sorted({str(column) for column in X.columns[X.columns.duplicated()].tolist()})
        raise PreprocessingError(f"WOE encoder schema contains duplicate columns: {duplicates}.")
    return X


def _jsonable_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _category_key(value: Any) -> str:
    return repr(_jsonable_scalar(value))


def _validate_binary_target(y_train: Any, *, expected_rows: int) -> pd.Series:
    if y_train is None:
        raise PreprocessingError("WOE encoder fit() requires y_train with binary target values.")
    y = pd.Series(y_train)
    if len(y) != expected_rows:
        raise PreprocessingError(
            f"WOE encoder X/y length mismatch: X has {expected_rows} rows, y has {len(y)} rows."
        )
    if y.isna().any():
        raise PreprocessingError("WOE encoder target contains missing values.")
    try:
        y_int = y.astype("int8")
    except ValueError as exc:
        raise PreprocessingError("WOE encoder target must contain binary values 0 and 1.") from exc
    domain = set(int(value) for value in y_int.unique())
    if domain != {0, 1}:
        raise PreprocessingError(f"WOE encoder target must contain exactly classes 0 and 1, got {sorted(domain)}.")
    return y_int.reset_index(drop=True)


class WeightOfEvidenceEncoder:
    """Encode categorical features as log(smoothed good distribution / bad distribution)."""

    def __init__(
        self,
        *,
        categorical_features: list[str] | tuple[str, ...],
        passthrough_features: list[str] | tuple[str, ...] = (),
        smoothing: float = 0.5,
        unknown_value: float = 0.0,
        sign_convention: str = WOE_SIGN_CONVENTION,
    ) -> None:
        if smoothing <= 0:
            raise PreprocessingError(f"WOE smoothing must be > 0, got {smoothing}.")
        if sign_convention != WOE_SIGN_CONVENTION:
            raise PreprocessingError(f"Unsupported WOE sign convention {sign_convention!r}.")
        self.categorical_features = list(categorical_features)
        self.passthrough_features = list(passthrough_features)
        self.smoothing = float(smoothing)
        self.unknown_value = float(unknown_value)
        self.sign_convention = sign_convention
        self._fitted = False
        self._input_feature_order: list[str] = []
        self._output_feature_order: list[str] = []
        self._mappings: dict[str, dict[Any, float]] = {}
        self._metadata_mappings: dict[str, dict[str, float]] = {}
        self._category_counts: dict[str, dict[str, dict[str, int]]] = {}
        self._class_counts: dict[str, int] = {}

    def _validate_schema(self, X: pd.DataFrame) -> None:
        columns = [str(column) for column in X.columns]
        if columns != self._input_feature_order:
            if set(columns) != set(self._input_feature_order):
                missing = sorted(set(self._input_feature_order) - set(columns))
                extra = sorted(set(columns) - set(self._input_feature_order))
                raise PreprocessingError(f"WOE transform schema mismatch; missing={missing}, extra={extra}.")
            raise PreprocessingError("WOE transform feature column order does not match fitted schema.")

    def fit(self, X_train: pd.DataFrame, y_train: Any) -> "WeightOfEvidenceEncoder":
        """Fit WOE mappings from training features and target only."""

        X = _ensure_dataframe(X_train)
        y = _validate_binary_target(y_train, expected_rows=len(X))
        columns = [str(column) for column in X.columns]
        configured = set(self.categorical_features) | set(self.passthrough_features)
        missing = sorted(configured - set(columns))
        extra = sorted(set(columns) - configured)
        if missing:
            raise PreprocessingError(f"WOE configured features missing from DataFrame: {missing}.")
        if extra:
            raise PreprocessingError(f"WOE input contains features not configured for encoding/passthrough: {extra}.")

        self._input_feature_order = columns
        self._output_feature_order = columns
        total_good = int((y == 0).sum())
        total_bad = int((y == 1).sum())
        self._class_counts = {"good": total_good, "bad": total_bad}
        self._mappings = {}
        self._metadata_mappings = {}
        self._category_counts = {}

        for feature in self.categorical_features:
            series = X[feature].reset_index(drop=True)
            categories = sorted(series.dropna().unique().tolist(), key=lambda value: (type(value).__name__, repr(value)))
            if not categories:
                raise PreprocessingError(f"WOE categorical feature {feature!r} has no non-missing values.")
            effective = len(categories)
            denominator_good = total_good + self.smoothing * effective
            denominator_bad = total_bad + self.smoothing * effective
            mapping: dict[Any, float] = {}
            metadata_mapping: dict[str, float] = {}
            counts: dict[str, dict[str, int]] = {}
            for category in categories:
                mask = series == category
                good_count = int(((y == 0) & mask).sum())
                bad_count = int(((y == 1) & mask).sum())
                good_distribution = (good_count + self.smoothing) / denominator_good
                bad_distribution = (bad_count + self.smoothing) / denominator_bad
                value = float(np.log(good_distribution / bad_distribution))
                if not np.isfinite(value):
                    raise PreprocessingError(f"WOE value for feature {feature!r}, category {category!r} is not finite.")
                mapping[_jsonable_scalar(category)] = value
                metadata_mapping[_category_key(category)] = value
                counts[_category_key(category)] = {"good": good_count, "bad": bad_count}
            self._mappings[feature] = mapping
            self._metadata_mappings[feature] = metadata_mapping
            self._category_counts[feature] = counts

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted WOE mappings without learning new categories."""

        transformed, _ = self.transform_with_diagnostics(X)
        return transformed

    def transform_with_diagnostics(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Transform and return unseen-category diagnostics without mutating state."""

        if not self._fitted:
            raise PreprocessingError("WeightOfEvidenceEncoder.transform() called before fit().")
        frame = _ensure_dataframe(X).copy(deep=True)
        self._validate_schema(frame)
        unseen_counts: dict[str, int] = {}
        output = pd.DataFrame(index=frame.index)

        for feature in self._output_feature_order:
            if feature in self._mappings:
                mapping = self._mappings[feature]
                unseen = ~frame[feature].isin(set(mapping))
                unseen_counts[feature] = int(unseen.sum())
                output[feature] = frame[feature].map(mapping).fillna(self.unknown_value).astype(float)
            else:
                values = pd.to_numeric(frame[feature], errors="raise").astype(float)
                if not np.isfinite(values.to_numpy()).all():
                    raise PreprocessingError(f"WOE passthrough feature {feature!r} contains non-finite values.")
                output[feature] = values

        diagnostics = {
            "unseen_category_counts": unseen_counts,
            "unseen_category_total": int(sum(unseen_counts.values())),
        }
        if not np.isfinite(output.to_numpy()).all():
            raise PreprocessingError("WOE output contains NaN or infinite values.")
        return output.loc[:, self._output_feature_order], diagnostics

    def fit_transform(self, X_train: pd.DataFrame, y_train: Any) -> pd.DataFrame:
        """Fit on training data and return encoded training features."""

        return self.fit(X_train, y_train).transform(X_train)

    def get_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable fitted WOE state."""

        return {
            "transformer": "WeightOfEvidenceEncoder",
            "version": WOE_VERSION,
            "fitted": self._fitted,
            "feature_order": list(self._input_feature_order),
            "encoded_features": list(self.categorical_features),
            "passthrough_numeric_features": list(self.passthrough_features),
            "target_mapping": {"0": "good/non-default", "1": "bad/default"},
            "sign_convention": self.sign_convention,
            "formula": "ln(smoothed_good_distribution / smoothed_bad_distribution)",
            "smoothing": self.smoothing,
            "unknown_fallback": self.unknown_value,
            "class_counts": dict(self._class_counts),
            "category_counts": {
                feature: {category: dict(counts) for category, counts in values.items()}
                for feature, values in self._category_counts.items()
            },
            "woe_mapping": {
                feature: dict(mapping) for feature, mapping in self._metadata_mappings.items()
            },
            "output_feature_order": list(self._output_feature_order),
        }
