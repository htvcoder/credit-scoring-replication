"""Atomic writer for P2B split artifacts."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.git import get_git_provenance
from creditrep.artifacts.split_definition import validate_split_definition, write_split_csv
from creditrep.checksums import DatasetChecksum
from creditrep.config.loader import config_hash
from creditrep.config.models import ExperimentConfig
from creditrep.datasets.models import LoadedDataset
from creditrep.datasets.registry import find_repo_root, resolve_repo_path
from creditrep.splitting.models import SplitResult

SCHEMA_VERSION = "1.0"
PACKAGE_VERSION = "0.2.0"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "experiment"


def build_experiment_id(config: ExperimentConfig, split_hash: str, *, created_at_utc: datetime | None = None) -> str:
    timestamp = (created_at_utc or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{_slugify(config.experiment_name)}-{timestamp}-{split_hash[:8]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _split_hash_payload(
    *,
    dataset: LoadedDataset,
    split: SplitResult,
    checksum: DatasetChecksum,
    config: ExperimentConfig,
) -> dict[str, Any]:
    return {
        "dataset": {
            "checksum_sha256": checksum.actual_sha256,
            "id": dataset.dataset_id,
            "source_file": dataset.metadata["source_file"],
        },
        "split": {
            "random_seed": config.random_seed,
            "strategy": config.split_strategy,
            "test_indices": list(split.test_indices),
            "test_size": config.test_size,
            "train_indices": list(split.train_indices),
        },
    }


def build_manifest(
    *,
    experiment_id: str,
    config: ExperimentConfig,
    dataset: LoadedDataset,
    split: SplitResult,
    checksum: DatasetChecksum,
    repo_root: Path,
    created_at_utc: datetime,
) -> dict[str, Any]:
    provenance = get_git_provenance(repo_root).to_dict()
    provenance.update(
        {
            "config_hash": config_hash(config),
            "created_at_utc": created_at_utc.isoformat().replace("+00:00", "Z"),
            "package_version": PACKAGE_VERSION,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_name": config.experiment_name,
        "status": "split_created",
        "dataset": {
            "id": dataset.dataset_id.upper(),
            "source_file": dataset.metadata["source_file"],
            "checksum_sha256": checksum.actual_sha256,
            "checksum_declared_sha256": checksum.declared_sha256,
            "checksum_match": checksum.matches,
            "row_count": dataset.metadata["row_count"],
            "feature_count": dataset.metadata["feature_count"],
            "class_counts": {str(key): value for key, value in dataset.metadata["class_counts"].items()},
        },
        "split": {
            "strategy": config.split_strategy,
            "test_size": config.test_size,
            "random_seed": config.random_seed,
            "shuffle": config.shuffle,
            "row_index_contract": split.metadata["row_index_contract"],
            "train_row_count": split.metadata["train_row_count"],
            "test_row_count": split.metadata["test_row_count"],
            "train_class_counts": {str(key): value for key, value in split.metadata["train_class_counts"].items()},
            "test_class_counts": {str(key): value for key, value in split.metadata["test_class_counts"].items()},
            "split_hash": split.split_hash,
            "split_definition": "split.csv",
        },
        "provenance": provenance,
        "reserved_artifacts": {
            "metrics": None,
            "predictions": None,
            "trained_model": None,
            "plots": None,
        },
    }


def create_split_artifact(
    *,
    config: ExperimentConfig,
    dataset: LoadedDataset,
    split: SplitResult,
    checksum: DatasetChecksum,
    repo_root: Path | str | None = None,
    created_at_utc: datetime | None = None,
) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    output_root = resolve_repo_path(config.output_root, repo_root=root, context="output.root_dir")
    created = created_at_utc or datetime.now(timezone.utc)
    experiment_id = build_experiment_id(config, split.split_hash, created_at_utc=created)
    final_dir = output_root / experiment_id
    temp_dir = output_root / f".tmp-{experiment_id}"
    if final_dir.exists():
        raise ArtifactError(f"Artifact directory already exists and will not be overwritten: {final_dir}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True)
    try:
        split_payload = _split_hash_payload(dataset=dataset, split=split, checksum=checksum, config=config)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            dataset=dataset,
            split=split,
            checksum=checksum,
            repo_root=root,
            created_at_utc=created,
        )
        _write_json(temp_dir / "manifest.json", manifest)
        _write_json(
            temp_dir / "split.json",
            {
                "schema_version": SCHEMA_VERSION,
                "row_index_contract": "row_position",
                "split_hash": split.split_hash,
                "split_hash_payload": split_payload,
                "row_count": split.metadata["row_count"],
                "train_row_count": split.metadata["train_row_count"],
                "test_row_count": split.metadata["test_row_count"],
                "definition_file": "split.csv",
            },
        )
        with (temp_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.canonical_payload(), handle, sort_keys=True, allow_unicode=True)
        write_split_csv(temp_dir / "split.csv", train_indices=split.train_indices, test_indices=split.test_indices)
        validate_split_definition(
            temp_dir / "split.csv",
            row_count=split.metadata["row_count"],
            expected_split_hash=split.split_hash,
            split_hash_payload=split_payload,
        )
        temp_dir.rename(final_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return final_dir


def create_smoke_experiment_artifact(
    *,
    config: ExperimentConfig,
    dataset: LoadedDataset,
    split: SplitResult,
    checksum: DatasetChecksum,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    model_metadata: dict[str, Any],
    prediction_hash: str,
    repo_root: Path | str | None = None,
    created_at_utc: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write a completed P2C smoke experiment artifact atomically."""

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    output_root = resolve_repo_path(config.output_root, repo_root=root, context="output.root_dir")
    created = created_at_utc or datetime.now(timezone.utc)
    experiment_id = build_experiment_id(config, split.split_hash, created_at_utc=created)
    final_dir = output_root / experiment_id
    temp_dir = output_root / f".tmp-{experiment_id}"
    if final_dir.exists():
        raise ArtifactError(f"Artifact directory already exists and will not be overwritten: {final_dir}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True)
    try:
        split_payload = _split_hash_payload(dataset=dataset, split=split, checksum=checksum, config=config)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            dataset=dataset,
            split=split,
            checksum=checksum,
            repo_root=root,
            created_at_utc=created,
        )
        manifest.update(
            {
                "status": "completed",
                "result_scope": "smoke_validation",
                "publishable": False,
                "model": {
                    "type": config.model_type,
                    "parameters": config.model_parameters,
                    "model_artifact_saved": False,
                },
                "preprocessing": {"mode": config.preprocessing_mode},
                "evaluation": {
                    "metrics_file": "metrics.json",
                    "predictions_file": "predictions.csv",
                    "model_metadata_file": "model_metadata.json",
                    "classification_threshold": config.classification_threshold,
                    "prediction_hash": prediction_hash,
                },
            }
        )
        manifest["reserved_artifacts"] = {
            "metrics": "metrics.json",
            "plots": None,
            "predictions": "predictions.csv",
            "trained_model": None,
        }
        _write_json(temp_dir / "manifest.json", manifest)
        _write_json(
            temp_dir / "split.json",
            {
                "schema_version": SCHEMA_VERSION,
                "row_index_contract": "row_position",
                "split_hash": split.split_hash,
                "split_hash_payload": split_payload,
                "row_count": split.metadata["row_count"],
                "train_row_count": split.metadata["train_row_count"],
                "test_row_count": split.metadata["test_row_count"],
                "definition_file": "split.csv",
            },
        )
        with (temp_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.canonical_payload(), handle, sort_keys=True, allow_unicode=True)
        write_split_csv(temp_dir / "split.csv", train_indices=split.train_indices, test_indices=split.test_indices)
        metrics_payload = {
            "schema_version": SCHEMA_VERSION,
            "model_type": config.model_type,
            "publishable": False,
            "result_scope": "smoke_validation",
            "split_hash": split.split_hash,
            "prediction_hash": prediction_hash,
            "threshold": config.classification_threshold,
            "test_metrics": metrics,
        }
        _write_json(temp_dir / "metrics.json", metrics_payload)
        _write_json(temp_dir / "model_metadata.json", model_metadata)
        predictions.to_csv(temp_dir / "predictions.csv", index=False, encoding="utf-8", float_format="%.17g")
        validate_split_definition(
            temp_dir / "split.csv",
            row_count=split.metadata["row_count"],
            expected_split_hash=split.split_hash,
            split_hash_payload=split_payload,
        )
        temp_dir.rename(final_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return final_dir, manifest
