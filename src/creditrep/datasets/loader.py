"""Load configured credit-scoring datasets through one P2A interface."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from creditrep.datasets.exceptions import DatasetFileError, DatasetSchemaError
from creditrep.datasets.models import DatasetSpec, LoadedDataset
from creditrep.datasets.registry import (
    find_repo_root,
    get_dataset_spec,
    load_registry,
    resolve_repo_path,
)


def _read_frame(spec: DatasetSpec, source: BytesIO) -> pd.DataFrame:
    reader_type = spec.reader.get("type")
    na_values = list(spec.missing_values)
    if reader_type == "csv":
        return pd.read_csv(
            source,
            na_values=na_values or None,
            encoding=spec.reader.get("encoding"),
        )
    if reader_type == "delimited":
        sep = (
            r"\s+"
            if spec.reader.get("sep") == "whitespace"
            else spec.reader.get("sep", ",")
        )
        header = None if spec.reader.get("header") is False else "infer"
        return pd.read_csv(
            source,
            sep=sep,
            header=header,
            names=spec.reader.get("columns"),
            na_values=na_values or None,
            encoding=spec.reader.get("encoding"),
        )
    if reader_type == "excel":
        return pd.read_excel(
            source,
            engine=spec.reader.get("engine", "xlrd"),
            header=spec.reader.get("header", 0),
            na_values=na_values or None,
        )
    raise DatasetFileError(
        f"{spec.dataset_id}: unsupported reader type {reader_type!r} for active file {spec.active_file}."
    )


def _target_key(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_target(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    if spec.target_column not in df.columns:
        raise DatasetSchemaError(
            f"{spec.dataset_id}: target column {spec.target_column!r} is missing from {spec.active_file}."
        )
    raw_target = df[spec.target_column]
    if raw_target.isna().any():
        raise DatasetSchemaError(
            f"{spec.dataset_id}: target column {spec.target_column!r} has null values after normalization."
        )
    target_keys = raw_target.map(_target_key)
    unknown = sorted(set(target_keys) - set(spec.target_mapping))
    if unknown:
        raise DatasetSchemaError(
            f"{spec.dataset_id}: target column {spec.target_column!r} in {spec.active_file} "
            f"contains values outside mapping: {unknown}."
        )
    normalized = target_keys.map(spec.target_mapping)
    if normalized.isna().any():
        raise DatasetSchemaError(
            f"{spec.dataset_id}: target column {spec.target_column!r} has null values after normalization."
        )
    normalized = normalized.astype("int8")
    domain = set(normalized.unique())
    if domain != {0, 1}:
        raise DatasetSchemaError(
            f"{spec.dataset_id}: normalized target must contain both binary classes 0 and 1, got {sorted(domain)}."
        )
    normalized.name = spec.target_column
    return normalized


def _validate_declared_columns(df: pd.DataFrame, spec: DatasetSpec) -> None:
    columns = set(df.columns)
    for group_name, declared in (
        ("identifier_columns", spec.identifier_columns),
        ("ignored_columns", spec.ignored_columns),
        ("categorical_columns", spec.categorical_columns),
        ("numeric_columns", spec.numeric_columns),
    ):
        missing = sorted(set(declared) - columns)
        if missing:
            raise DatasetSchemaError(
                f"{spec.dataset_id}: {group_name} declared missing columns in {spec.active_file}: {missing}."
            )


def _build_metadata(
    *,
    spec: DatasetSpec,
    features: pd.DataFrame,
    target: pd.Series,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    class_counts = {
        int(key): int(value)
        for key, value in target.value_counts().sort_index().items()
    }
    default_rate = float(class_counts.get(1, 0) / len(target)) if len(target) else 0.0
    removed_identifier_columns = [
        column
        for column in spec.identifier_columns
        if column in spec.raw.get("identifier_columns", [])
    ]
    removed_ignored_columns = [
        column
        for column in spec.ignored_columns
        if column not in removed_identifier_columns
    ]
    try:
        absolute_source = str(
            source_path.resolve().relative_to(repo_root.resolve())
        ).replace("\\", "/")
    except ValueError:
        absolute_source = spec.active_file
    return {
        "dataset_id": spec.dataset_id,
        "source_file": spec.active_file,
        "source_path": absolute_source,
        "target_column": spec.target_column,
        "target_mapping": {key: value for key, value in spec.target_mapping.items()},
        "removed_identifier_columns": removed_identifier_columns,
        "removed_ignored_columns": removed_ignored_columns,
        "removed_columns": list(
            dict.fromkeys([*removed_identifier_columns, *removed_ignored_columns])
        ),
        "row_count": int(len(target)),
        "feature_count": int(features.shape[1]),
        "class_counts": class_counts,
        "default_rate": default_rate,
        "categorical_columns": list(spec.categorical_columns),
        "numeric_columns": list(spec.numeric_columns),
    }


def load_dataset(
    dataset_id: str,
    *,
    registry_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    registry: dict[str, DatasetSpec] | None = None,
    expected_source_sha256: str | None = None,
) -> LoadedDataset:
    """Load a dataset as features, normalized binary target and metadata."""

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    selected_registry = (
        registry
        if registry is not None
        else load_registry(registry_path, repo_root=root)
    )
    spec = get_dataset_spec(dataset_id, selected_registry)
    source_path = resolve_repo_path(
        spec.active_file, repo_root=root, context=f"{spec.dataset_id}.active_file"
    )
    if not source_path.exists():
        raise DatasetFileError(
            f"{spec.dataset_id}: active file does not exist: {spec.active_file}"
        )
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise DatasetFileError(
            f"{spec.dataset_id}: cannot read active file: {spec.active_file}"
        ) from exc
    source_digest = sha256(source_bytes).hexdigest().upper()
    if (
        expected_source_sha256 is not None
        and source_digest != expected_source_sha256.upper()
    ):
        raise DatasetFileError(
            f"{spec.dataset_id}: source changed after runtime-input authorization."
        )

    df = _read_frame(spec, BytesIO(source_bytes))
    _validate_declared_columns(df, spec)
    target = _normalize_target(df, spec)
    remove_columns = list(
        dict.fromkeys(
            [spec.target_column, *spec.identifier_columns, *spec.ignored_columns]
        )
    )
    features = df.drop(columns=remove_columns)
    if features.shape[1] == 0:
        raise DatasetSchemaError(
            f"{spec.dataset_id}: no feature columns remain after removing target/identifier columns."
        )
    metadata = _build_metadata(
        spec=spec,
        features=features,
        target=target,
        source_path=source_path,
        repo_root=root,
    )
    return LoadedDataset(
        dataset_id=spec.dataset_id,
        features=features,
        target=target,
        metadata=metadata,
        source_path=source_path,
    )
