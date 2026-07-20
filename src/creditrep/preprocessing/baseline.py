"""Leakage-safe smoke baseline preprocessing."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _columns_from_metadata(features_columns: list[str], metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    columns = set(features_columns)
    numeric = [column for column in metadata.get("numeric_columns", []) if column in columns]
    categorical = [column for column in metadata.get("categorical_columns", []) if column in columns]
    declared = set(numeric) | set(categorical)
    remaining = [column for column in features_columns if column not in declared]
    numeric.extend(remaining)
    return numeric, categorical


def build_smoke_preprocessor(
    *,
    features_columns: list[str],
    dataset_metadata: dict[str, Any],
    model_type: str,
) -> ColumnTransformer:
    """Build a train-fitted-only sklearn preprocessor for P2C smoke runs."""

    numeric_columns, categorical_columns = _columns_from_metadata(features_columns, dataset_metadata)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
