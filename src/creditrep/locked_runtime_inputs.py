"""Semantic lock for repository inputs consumed by target outer-refit workers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.models import DatasetSpec
from creditrep.datasets.registry import (
    find_repo_root,
    parse_dataset_spec,
    resolve_repo_path,
    read_repo_file_no_symlinks,
    validate_portable_path,
)
from creditrep.preprocessing import load_protocol_a_config
from creditrep.preprocessing.protocol import ProtocolAConfig
from creditrep.strict_yaml import StrictYAMLError, load_strict_yaml


LOCKED_RUNTIME_INPUT_SCHEMA_VERSION = 1
LOCKED_RUNTIME_DATASETS = ("AC", "GMC")
OUTER_PROJECTION_RUNTIME_DATASETS = ("AC", "GC", "TH02", "HMEQ", "TC", "GMC")
SHA256_RE = re.compile(r"^[A-Fa-f0-9]{64}$")


class LockedRuntimeInputError(ValueError):
    """Raised when a target runtime input cannot be locked or revalidated."""


@dataclass(frozen=True)
class ValidatedRuntimeInputs:
    """Validated semantic snapshot plus the exact typed values workers consume."""

    snapshot: dict[str, Any]
    digest: str
    protocol_config: ProtocolAConfig
    registry: dict[str, DatasetSpec]
    source_hashes: dict[str, str]


def _selected_registry(
    root: Path, dataset_ids: tuple[str, ...]
) -> dict[str, DatasetSpec]:
    path = root / "data" / "datasets.yaml"
    if not path.is_file():
        raise LockedRuntimeInputError("dataset_registry_missing")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = load_strict_yaml(handle)
    except (OSError, StrictYAMLError) as exc:
        raise LockedRuntimeInputError("dataset_registry_invalid") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise LockedRuntimeInputError("dataset_registry_schema_mismatch")
    raw_datasets = payload["datasets"]
    specs: dict[str, DatasetSpec] = {}
    for dataset_id in dataset_ids:
        raw = raw_datasets.get(dataset_id.lower())
        if not isinstance(raw, dict):
            raise LockedRuntimeInputError(f"{dataset_id}: registry_entry_missing")
        _validate_selected_raw_schema(dataset_id, raw)
        specs[dataset_id.lower()] = parse_dataset_spec(dataset_id, raw)
    return specs


def _validate_selected_raw_schema(dataset_id: str, raw: dict[str, Any]) -> None:
    """Reject coercible types for every selected-registry value the loader uses."""

    for key in ("id", "active_file", "raw_file", "processed_file"):
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise LockedRuntimeInputError(f"{dataset_id}: {key}_invalid")
    target = raw.get("target")
    if not isinstance(target, dict):
        raise LockedRuntimeInputError(f"{dataset_id}: target_invalid")
    column = target.get("column") or raw.get("target_column")
    mapping = target.get("mapping_to_binary") or raw.get("target_mapping")
    if not isinstance(column, str) or not column or not isinstance(mapping, dict):
        raise LockedRuntimeInputError(f"{dataset_id}: target_invalid")
    if not mapping or any(
        type(key) not in {str, int} or type(value) is not int or value not in {0, 1}
        for key, value in mapping.items()
    ):
        raise LockedRuntimeInputError(f"{dataset_id}: target_mapping_invalid")
    for key in (
        "identifier_columns",
        "ignored_columns",
        "categorical_columns",
        "numeric_columns",
    ):
        value = raw.get(key, [])
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise LockedRuntimeInputError(f"{dataset_id}: {key}_invalid")
    missing = raw.get("missing_values", [])
    if missing is not None and (
        not isinstance(missing, list)
        or any(
            value is not None and type(value) not in {str, int, float}
            for value in missing
        )
        or any(type(value) is float and not math.isfinite(value) for value in missing)
    ):
        raise LockedRuntimeInputError(f"{dataset_id}: missing_values_invalid")
    reader = raw.get("reader")
    if not isinstance(reader, dict) or not isinstance(reader.get("type"), str):
        raise LockedRuntimeInputError(f"{dataset_id}: reader_invalid")


def _reader_projection(spec: DatasetSpec) -> dict[str, Any]:
    reader_type = spec.reader.get("type")
    if reader_type == "csv":
        encoding = spec.reader.get("encoding")
        if encoding is not None and not isinstance(encoding, str):
            raise LockedRuntimeInputError(
                f"{spec.dataset_id}: reader.encoding must be a string or null"
            )
        return {
            "type": "csv",
            "encoding": encoding,
        }
    if reader_type == "delimited":
        columns = spec.reader.get("columns")
        if columns is not None and (
            not isinstance(columns, list)
            or any(not isinstance(column, str) or not column for column in columns)
        ):
            raise LockedRuntimeInputError(
                f"{spec.dataset_id}: reader.columns must be a list or null"
            )
        header = spec.reader.get("header")
        if header is not None and type(header) is not bool:
            raise LockedRuntimeInputError(
                f"{spec.dataset_id}: reader.header must be boolean or null"
            )
        separator = spec.reader.get("sep", ",")
        encoding = spec.reader.get("encoding")
        if not isinstance(separator, str) or (
            encoding is not None and not isinstance(encoding, str)
        ):
            raise LockedRuntimeInputError(
                f"{spec.dataset_id}: reader separator/encoding invalid"
            )
        return {
            "type": "delimited",
            "sep": separator,
            "header": header,
            "columns": columns,
            "encoding": encoding,
        }
    if reader_type == "excel":
        header = spec.reader.get("header", 0)
        engine = spec.reader.get("engine", "xlrd")
        if not isinstance(header, int) or isinstance(header, bool) or header < 0:
            raise LockedRuntimeInputError(
                f"{spec.dataset_id}: reader.header must be a non-negative integer"
            )
        if not isinstance(engine, str) or not engine:
            raise LockedRuntimeInputError(
                f"{spec.dataset_id}: reader.engine must be a non-empty string"
            )
        return {
            "type": "excel",
            "engine": engine,
            "header": header,
        }
    raise LockedRuntimeInputError(
        f"{spec.dataset_id}: unsupported reader type {reader_type!r}"
    )


def _dataset_projection(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "dataset_id": spec.dataset_id,
        "active_file": spec.active_file,
        "target_column": spec.target_column,
        "target_mapping": dict(sorted(spec.target_mapping.items())),
        "identifier_columns": list(spec.identifier_columns),
        "ignored_columns": list(spec.ignored_columns),
        "categorical_columns": list(spec.categorical_columns),
        "numeric_columns": list(spec.numeric_columns),
        "missing_values": list(spec.missing_values),
        "reader": _reader_projection(spec),
    }


def _selected_checksums(
    root: Path, specs: dict[str, DatasetSpec], dataset_ids: tuple[str, ...]
) -> tuple[list[dict[str, str]], dict[str, str]]:
    path = root / "data" / "checksums-sha256.csv"
    if not path.is_file():
        raise LockedRuntimeInputError("checksum_registry_missing")
    selected_paths = {spec.active_file for spec in specs.values()}
    matches: dict[str, str] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["Path", "Algorithm", "Hash"]:
                raise LockedRuntimeInputError("checksum_registry_schema_mismatch")
            for row in reader:
                rel_path = row.get("Path")
                if rel_path not in selected_paths:
                    continue
                validate_portable_path(rel_path, context="checksums-sha256.csv")
                if rel_path in matches:
                    raise LockedRuntimeInputError("duplicate_selected_checksum_row")
                digest = row.get("Hash")
                if (
                    row.get("Algorithm", "").upper() != "SHA256"
                    or not isinstance(digest, str)
                    or not SHA256_RE.fullmatch(digest)
                ):
                    raise LockedRuntimeInputError("selected_checksum_row_invalid")
                matches[rel_path] = digest.upper()
    except (OSError, csv.Error) as exc:
        raise LockedRuntimeInputError("checksum_registry_invalid") from exc
    if set(matches) != selected_paths:
        raise LockedRuntimeInputError("selected_checksum_row_missing")

    rows: list[dict[str, str]] = []
    source_hashes: dict[str, str] = {}
    for dataset_id in dataset_ids:
        spec = specs[dataset_id.lower()]
        source_path = resolve_repo_path(
            spec.active_file, repo_root=root, context=f"{dataset_id}.active_file"
        )
        try:
            source_bytes = read_repo_file_no_symlinks(
                spec.active_file,
                repo_root=root,
                context=f"{dataset_id}.active_file",
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            code = (
                "source_file_missing"
                if not source_path.exists()
                else "source_symlink_or_path_invalid"
            )
            raise LockedRuntimeInputError(f"{dataset_id}: {code}") from exc
        actual = hashlib.sha256(source_bytes).hexdigest().upper()
        if actual != matches[spec.active_file]:
            raise LockedRuntimeInputError(f"{dataset_id}: source_checksum_mismatch")
        source_hashes[dataset_id] = actual
        rows.append(
            {
                "dataset_id": dataset_id,
                "path": spec.active_file,
                "algorithm": "SHA256",
                "sha256": actual,
            }
        )
    return rows, source_hashes


def load_locked_runtime_inputs(
    repo_root: Path | str | None = None,
    *,
    dataset_ids: tuple[str, ...] = LOCKED_RUNTIME_DATASETS,
) -> ValidatedRuntimeInputs:
    """Load, semantically project and verify all target workload runtime inputs."""

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    try:
        protocol_config = load_protocol_a_config(repo_root=root)
        if (
            not isinstance(dataset_ids, tuple)
            or not dataset_ids
            or len(dataset_ids) != len(set(dataset_ids))
            or any(
                not isinstance(dataset_id, str)
                or dataset_id not in OUTER_PROJECTION_RUNTIME_DATASETS
                for dataset_id in dataset_ids
            )
        ):
            raise LockedRuntimeInputError("locked_runtime_dataset_inventory_invalid")
        specs = _selected_registry(root, dataset_ids)
        checksum_rows, source_hashes = _selected_checksums(root, specs, dataset_ids)
        snapshot = {
            "schema_version": LOCKED_RUNTIME_INPUT_SCHEMA_VERSION,
            "protocol_a": asdict(protocol_config),
            "datasets": {
                dataset_id: _dataset_projection(specs[dataset_id.lower()])
                for dataset_id in dataset_ids
            },
            "selected_checksums": checksum_rows,
        }
        digest = sha256_canonical(snapshot).upper()
    except LockedRuntimeInputError:
        raise
    except Exception as exc:
        raise LockedRuntimeInputError("locked_runtime_input_invalid") from exc
    return ValidatedRuntimeInputs(
        snapshot=snapshot,
        digest=digest,
        protocol_config=protocol_config,
        registry=specs,
        source_hashes=source_hashes,
    )


def validate_locked_runtime_inputs(
    expected_digest: Any,
    repo_root: Path | str | None = None,
    *,
    dataset_ids: tuple[str, ...] = LOCKED_RUNTIME_DATASETS,
) -> ValidatedRuntimeInputs:
    """Reload current inputs and compare them with the authorized semantic digest."""

    if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(expected_digest):
        raise LockedRuntimeInputError("locked_runtime_input_digest_invalid")
    value = load_locked_runtime_inputs(repo_root, dataset_ids=dataset_ids)
    if value.digest != expected_digest.upper():
        raise LockedRuntimeInputError("locked_runtime_input_mismatch")
    return value
