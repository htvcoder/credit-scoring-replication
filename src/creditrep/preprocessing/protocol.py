"""Leakage-safe Protocol A preprocessing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from creditrep.preprocessing.exceptions import PreprocessingError

PROTOCOL_A_NAME = "protocol_a"
PROTOCOL_A_VERSION = "0.1.0"
UNKNOWN_CATEGORY_TOKEN = "__UNKNOWN__"
SUPPORTED_NUMERIC_IMPUTATION = {"mean"}
SUPPORTED_CATEGORICAL_IMPUTATION = {"most_frequent", "mode"}
SUPPORTED_UNSEEN_CATEGORY = {"reserved_token"}


class PreprocessingProtocol(Protocol):
    """Minimal fit/transform contract for train-only preprocessing."""

    def fit(self, X_train: pd.DataFrame, y_train: Any | None = None) -> "PreprocessingProtocol":
        """Fit preprocessing state from training features only."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features without mutating fitted state."""

    def transform_with_diagnostics(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Transform features and return non-state diagnostics."""

    def fit_transform(self, X_train: pd.DataFrame, y_train: Any | None = None) -> pd.DataFrame:
        """Fit on training features and return transformed training features."""

    def get_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable view of fitted state."""


@dataclass(frozen=True)
class ProtocolAConfig:
    """Validated configuration for Protocol A preprocessing."""

    protocol_name: str = PROTOCOL_A_NAME
    numeric_imputation_strategy: str = "mean"
    categorical_imputation_strategy: str = "most_frequent"
    unseen_category_strategy: str = "reserved_token"
    unknown_token: str = UNKNOWN_CATEGORY_TOKEN


def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise PreprocessingError(f"Preprocessor expects a pandas DataFrame, got {type(X).__name__}.")
    if X.columns.has_duplicates:
        duplicates = sorted({str(column) for column in X.columns[X.columns.duplicated()].tolist()})
        raise PreprocessingError(f"Input feature schema contains duplicate columns: {duplicates}.")
    return X


def _validate_protocol_config(config: ProtocolAConfig) -> None:
    if config.protocol_name != PROTOCOL_A_NAME:
        raise PreprocessingError(f"Unsupported preprocessing protocol {config.protocol_name!r}.")
    if config.numeric_imputation_strategy not in SUPPORTED_NUMERIC_IMPUTATION:
        raise PreprocessingError(
            f"Unsupported numeric imputation strategy {config.numeric_imputation_strategy!r}."
        )
    if config.categorical_imputation_strategy not in SUPPORTED_CATEGORICAL_IMPUTATION:
        raise PreprocessingError(
            f"Unsupported categorical imputation strategy {config.categorical_imputation_strategy!r}."
        )
    if config.unseen_category_strategy not in SUPPORTED_UNSEEN_CATEGORY:
        raise PreprocessingError(f"Unsupported unseen-category strategy {config.unseen_category_strategy!r}.")
    if not isinstance(config.unknown_token, str) or not config.unknown_token:
        raise PreprocessingError("Unknown category token must be a non-empty string.")


def _metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise PreprocessingError(f"dataset metadata field {key!r} must be a list.")
    return [str(item) for item in value]


def features_from_dataset_metadata(
    features_columns: list[str],
    dataset_metadata: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Route features with registry metadata and reject mismatches."""

    numeric = _metadata_list(dataset_metadata, "numeric_columns")
    categorical = _metadata_list(dataset_metadata, "categorical_columns")
    target = dataset_metadata.get("target_column")
    removed = set(_metadata_list(dataset_metadata, "removed_columns"))
    removed.update(_metadata_list(dataset_metadata, "removed_identifier_columns"))
    removed.update(_metadata_list(dataset_metadata, "removed_ignored_columns"))

    feature_set = set(features_columns)
    numeric_set = set(numeric)
    categorical_set = set(categorical)
    overlap = sorted(numeric_set & categorical_set)
    if overlap:
        raise PreprocessingError(f"Feature metadata cannot route columns as both numeric and categorical: {overlap}.")
    if target is not None and str(target) in feature_set:
        raise PreprocessingError(f"Target column {target!r} must not be present in preprocessing features.")
    removed_present = sorted(feature_set & removed)
    if removed_present:
        raise PreprocessingError(f"Removed identifier/ignored columns must not be preprocessed: {removed_present}.")

    declared = numeric_set | categorical_set
    missing = sorted(declared - feature_set)
    extra = sorted(feature_set - declared)
    if missing:
        raise PreprocessingError(f"Dataset metadata declares feature columns missing from DataFrame: {missing}.")
    if extra:
        raise PreprocessingError(f"DataFrame contains feature columns not declared in dataset metadata: {extra}.")

    numeric_ordered = [column for column in features_columns if column in numeric_set]
    categorical_ordered = [column for column in features_columns if column in categorical_set]
    return numeric_ordered, categorical_ordered


def _categorical_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _deterministic_tie_break(values: list[Any]) -> Any:
    return sorted(values, key=_categorical_sort_key)[0]


def _jsonable_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _jsonable_values(values: list[Any]) -> list[Any]:
    return [_jsonable_scalar(value) for value in values]


def _contains_unknown_token(series: pd.Series, token: str) -> bool:
    return series.map(lambda value: isinstance(value, str) and value == token).any()


class LeakageSafePreprocessor:
    """Protocol A mean/mode imputer with train-fitted categorical vocabulary.

    The transformer returns a DataFrame with the fitted column order. Calling
    ``transform`` never updates means, modes, categorical vocabularies or
    metadata used for reproducibility. Per-transform diagnostics are returned
    separately by ``transform_with_diagnostics``.
    """

    def __init__(
        self,
        *,
        numeric_features: list[str] | tuple[str, ...] | None = None,
        categorical_features: list[str] | tuple[str, ...] | None = None,
        dataset_metadata: dict[str, Any] | None = None,
        config: ProtocolAConfig | None = None,
    ) -> None:
        self.config = config or ProtocolAConfig()
        _validate_protocol_config(self.config)
        self._provided_numeric = list(numeric_features) if numeric_features is not None else None
        self._provided_categorical = list(categorical_features) if categorical_features is not None else None
        self._dataset_metadata = dict(dataset_metadata) if dataset_metadata is not None else None
        self._fitted = False
        self._numeric_features: list[str] = []
        self._categorical_features: list[str] = []
        self._input_feature_order: list[str] = []
        self._numeric_imputation_values: dict[str, float] = {}
        self._categorical_imputation_values: dict[str, Any] = {}
        self._category_vocabulary: dict[str, list[Any]] = {}
        self._fitted_row_count = 0

    def _resolve_feature_routes(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        columns = [str(column) for column in X.columns]
        if self._dataset_metadata is not None:
            return features_from_dataset_metadata(columns, self._dataset_metadata)
        numeric = list(self._provided_numeric or [])
        categorical = list(self._provided_categorical or [])
        declared = set(numeric) | set(categorical)
        overlap = sorted(set(numeric) & set(categorical))
        if overlap:
            raise PreprocessingError(f"Columns cannot be both numeric and categorical: {overlap}.")
        missing = sorted(declared - set(columns))
        extra = sorted(set(columns) - declared)
        if missing:
            raise PreprocessingError(f"Configured preprocessing features missing from DataFrame: {missing}.")
        if extra:
            raise PreprocessingError(f"DataFrame contains columns not configured for preprocessing: {extra}.")
        return [column for column in columns if column in set(numeric)], [
            column for column in columns if column in set(categorical)
        ]

    def _validate_schema(self, X: pd.DataFrame) -> None:
        columns = [str(column) for column in X.columns]
        if columns != self._input_feature_order:
            if set(columns) != set(self._input_feature_order):
                missing = sorted(set(self._input_feature_order) - set(columns))
                extra = sorted(set(columns) - set(self._input_feature_order))
                raise PreprocessingError(
                    f"Transform feature schema does not match fitted schema; missing={missing}, extra={extra}."
                )
            raise PreprocessingError("Transform feature column order does not match fitted schema.")

    def fit(self, X_train: pd.DataFrame, y_train: Any | None = None) -> "LeakageSafePreprocessor":
        """Fit means, modes and vocabularies using training features only."""

        X = _ensure_dataframe(X_train)
        numeric, categorical = self._resolve_feature_routes(X)
        self._input_feature_order = [str(column) for column in X.columns]
        self._numeric_features = numeric
        self._categorical_features = categorical
        self._numeric_imputation_values = {}
        self._categorical_imputation_values = {}
        self._category_vocabulary = {}

        for column in numeric:
            values = pd.to_numeric(X[column], errors="coerce")
            non_missing = values.dropna()
            if non_missing.empty:
                raise PreprocessingError(f"Numeric feature {column!r} is all missing in training data.")
            self._numeric_imputation_values[column] = float(non_missing.mean())

        for column in categorical:
            series = X[column]
            non_missing = series[~series.isna()]
            if non_missing.empty:
                raise PreprocessingError(f"Categorical feature {column!r} is all missing in training data.")
            if _contains_unknown_token(non_missing, self.config.unknown_token):
                raise PreprocessingError(
                    f"Categorical feature {column!r} contains reserved unknown token "
                    f"{self.config.unknown_token!r} in training data."
                )
            counts = non_missing.value_counts(sort=False, dropna=True)
            max_count = counts.max()
            candidates = counts[counts == max_count].index.tolist()
            mode = _deterministic_tie_break(candidates)
            vocabulary = _jsonable_values(sorted(non_missing.unique().tolist(), key=_categorical_sort_key))
            self._categorical_imputation_values[column] = _jsonable_scalar(mode)
            self._category_vocabulary[column] = vocabulary

        self._fitted_row_count = int(len(X))
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted train-only state without learning from transform data."""

        transformed, _ = self.transform_with_diagnostics(X)
        return transformed

    def transform_with_diagnostics(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply fitted state and return diagnostics without mutating state."""

        if not self._fitted:
            raise PreprocessingError("LeakageSafePreprocessor.transform() called before fit().")
        frame = _ensure_dataframe(X).copy(deep=True)
        self._validate_schema(frame)
        unseen_counts: dict[str, int] = {}

        for column, value in self._numeric_imputation_values.items():
            numeric = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = numeric.fillna(value)

        for column, mode in self._categorical_imputation_values.items():
            series = frame[column].astype("object").copy()
            missing = series.isna()
            if missing.any():
                series.loc[missing] = mode
            if _contains_unknown_token(series, self.config.unknown_token):
                raise PreprocessingError(
                    f"Categorical feature {column!r} contains reserved unknown token "
                    f"{self.config.unknown_token!r} during transform."
                )
            vocabulary = set(self._category_vocabulary[column])
            unseen = ~series.isna() & ~series.isin(vocabulary)
            unseen_counts[column] = int(unseen.sum())
            if unseen.any():
                series.loc[unseen] = self.config.unknown_token
            frame[column] = series

        diagnostics = {
            "unseen_category_counts": unseen_counts,
            "unseen_category_total": int(sum(unseen_counts.values())),
        }
        return frame.loc[:, self._input_feature_order], diagnostics

    def fit_transform(self, X_train: pd.DataFrame, y_train: Any | None = None) -> pd.DataFrame:
        """Fit on training features and return transformed training features."""

        return self.fit(X_train, y_train=y_train).transform(X_train)

    def get_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable fitted state for reproducibility."""

        return {
            "protocol_name": self.config.protocol_name,
            "protocol_version": PROTOCOL_A_VERSION,
            "implementation_version": PROTOCOL_A_VERSION,
            "fitted": self._fitted,
            "numeric_features": list(self._numeric_features),
            "categorical_features": list(self._categorical_features),
            "numeric_imputation_strategy": self.config.numeric_imputation_strategy,
            "numeric_imputation_values": dict(self._numeric_imputation_values),
            "categorical_imputation_strategy": self.config.categorical_imputation_strategy,
            "categorical_imputation_values": dict(self._categorical_imputation_values),
            "category_vocabulary": {key: list(value) for key, value in self._category_vocabulary.items()},
            "unseen_category_strategy": self.config.unseen_category_strategy,
            "unknown_token": self.config.unknown_token,
            "fitted_row_count": self._fitted_row_count,
            "input_feature_order": list(self._input_feature_order),
        }
