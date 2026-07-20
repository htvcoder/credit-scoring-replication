"""Read and validate the dataset registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from creditrep.datasets.exceptions import DatasetNotFoundError, RegistryError
from creditrep.datasets.models import DatasetSpec


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from an arbitrary current working directory."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "datasets.yaml").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def is_windows_absolute(path_text: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_text)) or path_text.startswith("\\\\")


def validate_portable_path(path_text: str, *, context: str) -> None:
    if not path_text:
        raise RegistryError(f"{context}: path must not be empty.")
    if "\\" in path_text:
        raise RegistryError(f"{context}: path must use '/' separators: {path_text}")
    if is_windows_absolute(path_text):
        raise RegistryError(f"{context}: Windows absolute path is not portable: {path_text}")
    path = Path(path_text)
    if path.is_absolute() or path_text.startswith("/"):
        raise RegistryError(f"{context}: POSIX absolute path is not portable: {path_text}")
    if ".." in path.parts:
        raise RegistryError(f"{context}: path must not contain '..': {path_text}")


def resolve_repo_path(path_text: str, *, repo_root: Path, context: str) -> Path:
    validate_portable_path(path_text, context=context)
    return repo_root / Path(path_text)


def _normalize_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip().lower()
    if not normalized:
        raise DatasetNotFoundError("Dataset ID must not be empty.")
    return normalized


def _normalize_mapping(mapping: dict[Any, Any], *, dataset_id: str) -> dict[str, int]:
    if not isinstance(mapping, dict) or not mapping:
        raise RegistryError(f"{dataset_id}: target mapping is missing or empty.")
    normalized: dict[str, int] = {}
    for raw_value, binary_value in mapping.items():
        try:
            normalized[str(raw_value)] = int(binary_value)
        except (TypeError, ValueError) as exc:
            raise RegistryError(
                f"{dataset_id}: target mapping value for {raw_value!r} must be 0 or 1."
            ) from exc
    mapped_values = set(normalized.values())
    if mapped_values != {0, 1}:
        raise RegistryError(
            f"{dataset_id}: target mapping must produce both classes 0 and 1, got {sorted(mapped_values)}."
        )
    return normalized


def _list_field(raw: dict[str, Any], key: str, *, dataset_id: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RegistryError(f"{dataset_id}: {key} must be a list.")
    return tuple(str(item) for item in value)


def _active_file(raw: dict[str, Any], *, dataset_id: str) -> str:
    active = raw.get("active_file")
    if not active:
        reader = raw.get("reader") or {}
        if raw.get("processed_file") and reader.get("verify_file") == "processed":
            active = raw.get("processed_file")
        else:
            active = raw.get("raw_file")
    if not active:
        raise RegistryError(f"{dataset_id}: active_file is required.")
    return str(active)


def parse_dataset_spec(dataset_id: str, raw: dict[str, Any]) -> DatasetSpec:
    if not isinstance(raw, dict):
        raise RegistryError(f"{dataset_id}: registry entry must be a mapping.")
    declared_id = _normalize_dataset_id(str(raw.get("id", dataset_id)))
    normalized_id = _normalize_dataset_id(dataset_id)
    if declared_id != normalized_id:
        raise RegistryError(f"{dataset_id}: id field {declared_id!r} does not match registry key.")

    active_file = _active_file(raw, dataset_id=normalized_id)
    validate_portable_path(active_file, context=f"{normalized_id}.active_file")
    for key in ("raw_file", "processed_file"):
        if raw.get(key):
            validate_portable_path(str(raw[key]), context=f"{normalized_id}.{key}")
    for path_text in raw.get("dictionary_files") or []:
        validate_portable_path(str(path_text), context=f"{normalized_id}.dictionary_files")

    target_cfg = raw.get("target")
    if not isinstance(target_cfg, dict):
        raise RegistryError(f"{normalized_id}: target metadata is required.")
    target_column = target_cfg.get("column") or raw.get("target_column")
    if not target_column:
        raise RegistryError(f"{normalized_id}: target column is required.")
    target_mapping = target_cfg.get("mapping_to_binary") or raw.get("target_mapping")
    mapping = _normalize_mapping(target_mapping, dataset_id=normalized_id)

    identifiers = _list_field(raw, "identifier_columns", dataset_id=normalized_id)
    ignored = _list_field(raw, "ignored_columns", dataset_id=normalized_id)
    categorical = _list_field(raw, "categorical_columns", dataset_id=normalized_id)
    numeric = _list_field(raw, "numeric_columns", dataset_id=normalized_id)
    target_set = {str(target_column)}
    identifier_like = set(identifiers) | set(ignored)
    feature_metadata = set(categorical) | set(numeric)
    if target_set & identifier_like:
        raise RegistryError(
            f"{normalized_id}: target column {target_column!r} cannot also be identifier/ignored."
        )
    if target_set & feature_metadata:
        raise RegistryError(f"{normalized_id}: target column {target_column!r} cannot be a feature.")
    overlap = set(categorical) & set(numeric)
    if overlap:
        raise RegistryError(f"{normalized_id}: columns cannot be both numeric and categorical: {sorted(overlap)}")
    id_feature_overlap = set(identifiers) & feature_metadata
    if id_feature_overlap:
        raise RegistryError(
            f"{normalized_id}: identifier columns cannot also be features: {sorted(id_feature_overlap)}"
        )

    reader = raw.get("reader")
    if not isinstance(reader, dict):
        raise RegistryError(f"{normalized_id}: reader metadata is required.")

    return DatasetSpec(
        dataset_id=normalized_id,
        raw=raw,
        active_file=active_file,
        target_column=str(target_column),
        target_mapping=mapping,
        identifier_columns=identifiers,
        ignored_columns=ignored,
        categorical_columns=categorical,
        numeric_columns=numeric,
        missing_values=tuple(raw.get("missing_values") or []),
        reader=reader,
    )


def load_registry(
    registry_path: Path | str | None = None,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, DatasetSpec]:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    path = Path(registry_path) if registry_path is not None else root / "data" / "datasets.yaml"
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise RegistryError(f"Dataset registry does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise RegistryError(f"Dataset registry YAML is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise RegistryError(f"Dataset registry is malformed: {path}; expected top-level 'datasets'.")

    specs: dict[str, DatasetSpec] = {}
    for dataset_id, raw in payload["datasets"].items():
        spec = parse_dataset_spec(str(dataset_id), raw)
        specs[spec.dataset_id] = spec
    if not specs:
        raise RegistryError(f"Dataset registry is empty: {path}")
    return specs


def get_dataset_spec(
    dataset_id: str,
    registry: dict[str, DatasetSpec],
) -> DatasetSpec:
    normalized_id = _normalize_dataset_id(dataset_id)
    try:
        return registry[normalized_id]
    except KeyError as exc:
        valid = ", ".join(sorted(registry))
        raise DatasetNotFoundError(f"Dataset ID {dataset_id!r} is not in registry. Valid IDs: {valid}") from exc
