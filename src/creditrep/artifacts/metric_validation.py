"""Phase 4 metric-validation artifact writer and validator."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.git import get_git_provenance
from creditrep.artifacts.writer import PACKAGE_VERSION, _slugify
from creditrep.checksums import DatasetChecksum
from creditrep.config.loader import sha256_canonical
from creditrep.config.metric_validation import MetricValidationConfig
from creditrep.datasets.models import LoadedDataset
from creditrep.datasets.registry import find_repo_root, resolve_repo_path
from creditrep.experiments.metric_validation import MetricValidationResult
from creditrep.preprocessing.protocol import ProtocolAConfig
from creditrep.splitting.exceptions import SplitError
from creditrep.splitting.hashing import split_hash
from creditrep.splitting.nested import NestedCVDefinition, validate_nested_cv_definition

METRIC_VALIDATION_SCHEMA_VERSION = "0.1.0"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_outer_csv(path: Path, definition: NestedCVDefinition) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["outer_fold_id", "row_position", "partition"])
        writer.writeheader()
        for outer in definition.outer_folds:
            for row in outer.train_indices:
                writer.writerow({"outer_fold_id": outer.outer_fold_id, "row_position": row, "partition": "train"})
            for row in outer.test_indices:
                writer.writerow({"outer_fold_id": outer.outer_fold_id, "row_position": row, "partition": "test"})


def build_metric_validation_experiment_id(
    config: MetricValidationConfig,
    nested_cv_hash: str,
    *,
    created_at_utc: datetime | None = None,
) -> str:
    timestamp = (created_at_utc or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{_slugify(config.experiment_name)}-{timestamp}-{nested_cv_hash[:8]}"


def metric_validation_config_hash(config: MetricValidationConfig) -> str:
    return sha256_canonical(config.canonical_payload())


def protocol_config_hash(protocol_config: ProtocolAConfig) -> str:
    return split_hash({"protocol_config": protocol_config.__dict__})


def _fold_metric_records(
    *,
    config: MetricValidationConfig,
    dataset_checksum: str,
    protocol_config: ProtocolAConfig,
    git_commit: str | None,
    result: MetricValidationResult,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    config_hash = metric_validation_config_hash(config)
    protocol_hash = protocol_config_hash(protocol_config)
    for outer in result.nested_cv.outer_folds:
        for metric in result.fold_metrics[outer.outer_fold_id]:
            records.append(
                {
                    "config_hash": config_hash,
                    "dataset_checksum": dataset_checksum,
                    "direction": metric.direction,
                    "exactness": metric.exactness,
                    "git_commit": git_commit,
                    "metric_id": metric.metric_id,
                    "metric_version": metric.metric_version,
                    "outer_fold_id": outer.outer_fold_id,
                    "parameters": metric.parameters,
                    "preprocessing_protocol_name": protocol_config.protocol_name,
                    "preprocessing_protocol_version": protocol_config.protocol_version,
                    "protocol_config_hash": protocol_hash,
                    "result_scope": "metric_validation",
                    "seed": outer.seed,
                    "split_hash": outer.split_hash,
                    "status": metric.status,
                    "value": metric.value,
                    "warnings": list(metric.warnings),
                }
            )
    return records


def create_metric_validation_artifact(
    *,
    config: MetricValidationConfig,
    protocol_config: ProtocolAConfig,
    dataset: LoadedDataset,
    checksum: DatasetChecksum,
    result: MetricValidationResult,
    repo_root: Path | str | None = None,
    created_at_utc: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    output_root = resolve_repo_path(config.output_root, repo_root=root, context="output.root_dir")
    created = created_at_utc or datetime.now(timezone.utc)
    experiment_id = build_metric_validation_experiment_id(config, result.nested_cv.nested_cv_hash, created_at_utc=created)
    final_dir = output_root / experiment_id
    temp_dir = output_root / f".tmp-{experiment_id}"
    if final_dir.exists():
        raise ArtifactError(f"Artifact directory already exists and will not be overwritten: {final_dir}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True)
    try:
        nested_dir = temp_dir / "nested_cv"
        nested_dir.mkdir()
        outer_dir = nested_dir / "outer"
        outer_dir.mkdir()
        provenance = get_git_provenance(root).to_dict()
        provenance.update(
            {
                "config_hash": metric_validation_config_hash(config),
                "created_at_utc": created.isoformat().replace("+00:00", "Z"),
                "package_version": PACKAGE_VERSION,
                "protocol_config_hash": protocol_config_hash(protocol_config),
            }
        )
        manifest = {
            "schema_version": METRIC_VALIDATION_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "experiment_name": config.experiment_name,
            "status": "completed",
            "result_scope": "metric_validation",
            "publishable": False,
            "dataset": {
                "id": dataset.dataset_id.upper(),
                "source_file": dataset.metadata["source_file"],
                "checksum_sha256": checksum.actual_sha256,
                "checksum_declared_sha256": checksum.declared_sha256,
                "checksum_match": checksum.matches,
                "row_count": dataset.metadata["row_count"],
                "feature_count": dataset.metadata["feature_count"],
            },
            "nested_cv": {
                "manifest": "nested_cv/manifest.json",
                "outer_folds": "nested_cv/outer_folds.json",
                "outer_folds_csv": "nested_cv/outer_folds.csv",
                "nested_cv_hash": result.nested_cv.nested_cv_hash,
                "outer_fold_count": len(result.nested_cv.outer_folds),
                "inner_fold_count": sum(len(outer.inner_folds) for outer in result.nested_cv.outer_folds),
            },
            "preprocessing": {
                "protocol_name": protocol_config.protocol_name,
                "protocol_version": protocol_config.protocol_version,
                "protocol_config_hash": protocol_config_hash(protocol_config),
            },
            "evaluation": {
                "fold_metrics_file": "fold_metrics.json",
                "metrics_summary_file": "metrics_summary.json",
                "metric_config_hash": result.metric_config_hash,
                "metric_ids": [metric.metric_id for metric in config.metrics],
                "validation_model": config.validation_model,
            },
            "provenance": provenance,
            "reserved_artifacts": {"metrics": "fold_metrics.json", "predictions": None, "trained_model": None, "plots": None},
        }
        _write_json(temp_dir / "manifest.json", manifest)
        with (temp_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.canonical_payload(), handle, sort_keys=True, allow_unicode=True)
        _write_json(temp_dir / "fold_metrics.json", {"records": _fold_metric_records(
            config=config,
            dataset_checksum=checksum.actual_sha256,
            protocol_config=protocol_config,
            git_commit=provenance.get("git_commit"),
            result=result,
        ), "schema_version": METRIC_VALIDATION_SCHEMA_VERSION})
        _write_json(temp_dir / "metrics_summary.json", {"metrics": result.metric_summary, "schema_version": METRIC_VALIDATION_SCHEMA_VERSION})
        _write_json(temp_dir / "prediction_summary.json", {"folds": result.prediction_summaries, "schema_version": METRIC_VALIDATION_SCHEMA_VERSION})
        _write_json(temp_dir / "nested_cv" / "manifest.json", manifest["nested_cv"] | {"schema_version": METRIC_VALIDATION_SCHEMA_VERSION})
        _write_json(temp_dir / "nested_cv" / "outer_folds.json", result.nested_cv.to_dict())
        _write_outer_csv(temp_dir / "nested_cv" / "outer_folds.csv", result.nested_cv)
        for outer in result.nested_cv.outer_folds:
            outer_path = outer_dir / outer.outer_fold_id
            inner_path = outer_path / "inner"
            inner_path.mkdir(parents=True)
            _write_json(outer_path / "split.json", outer.to_dict())
            _write_json(outer_path / "preprocessing.json", result.outer_preprocessing[outer.outer_fold_id])
            _write_json(outer_path / "tuning_summary.json", result.tuning_summaries[outer.outer_fold_id])
            _write_json(
                outer_path / "metrics.json",
                {
                    "metrics": [metric.to_dict() for metric in result.fold_metrics[outer.outer_fold_id]],
                    "outer_fold_id": outer.outer_fold_id,
                    "schema_version": METRIC_VALIDATION_SCHEMA_VERSION,
                    "split_hash": outer.split_hash,
                },
            )
            for inner in outer.inner_folds:
                fold_path = inner_path / inner.inner_fold_id
                fold_path.mkdir()
                _write_json(fold_path / "split.json", inner.to_dict())
                _write_json(fold_path / "preprocessing.json", result.inner_preprocessing[inner.inner_fold_id])
        validate_metric_validation_artifact(temp_dir, target=dataset.target)
        temp_dir.rename(final_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return final_dir, manifest


def load_metric_validation_artifact(artifact_dir: Path | str) -> dict[str, Any]:
    path = Path(artifact_dir)
    required = [
        path / "manifest.json",
        path / "config.yaml",
        path / "fold_metrics.json",
        path / "metrics_summary.json",
        path / "nested_cv" / "outer_folds.json",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise ArtifactError(f"Metric validation artifact missing required files: {missing}.")
    with (path / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (path / "fold_metrics.json").open(encoding="utf-8") as handle:
        fold_metrics = json.load(handle)
    with (path / "metrics_summary.json").open(encoding="utf-8") as handle:
        metrics_summary = json.load(handle)
    with (path / "nested_cv" / "outer_folds.json").open(encoding="utf-8") as handle:
        folds = json.load(handle)
    return {"manifest": manifest, "fold_metrics": fold_metrics, "metrics_summary": metrics_summary, "outer_folds": folds}


def validate_metric_validation_artifact(artifact_dir: Path | str, *, target) -> None:
    loaded = load_metric_validation_artifact(artifact_dir)
    manifest = loaded["manifest"]
    folds = loaded["outer_folds"]
    fold_metrics = loaded["fold_metrics"]
    if manifest.get("publishable") is not False:
        raise ArtifactError("Metric validation artifact must set publishable=false.")
    if manifest.get("result_scope") != "metric_validation":
        raise ArtifactError("Metric validation artifact result_scope must be metric_validation.")
    if manifest["reserved_artifacts"].get("predictions") is not None:
        raise ArtifactError("Metric validation artifact must not publish row-level predictions.")
    expected_hash = split_hash({key: value for key, value in folds.items() if key != "nested_cv_hash"})
    if expected_hash != folds.get("nested_cv_hash"):
        raise ArtifactError("Nested CV hash mismatch.")
    if manifest["nested_cv"]["nested_cv_hash"] != folds["nested_cv_hash"]:
        raise ArtifactError("Manifest nested_cv_hash mismatch.")
    for record in fold_metrics.get("records", []):
        if record.get("result_scope") != "metric_validation":
            raise ArtifactError("Fold metric record has invalid result_scope.")
        if record.get("status") != "valid" and record.get("value") is not None:
            raise ArtifactError("Unsupported/undefined/failed metrics must not store a numeric value.")
    from creditrep.splitting.nested import InnerFoldDefinition, NestedCVDefinition, OuterFoldDefinition

    outer_defs = []
    for outer in folds["outer_folds"]:
        inner_defs = tuple(
            InnerFoldDefinition(
                parent_outer_fold_id=item["parent_outer_fold_id"],
                inner_fold_id=item["inner_fold_id"],
                train_indices=tuple(item["train_indices"]),
                validation_indices=tuple(item["validation_indices"]),
                train_class_counts={int(k): int(v) for k, v in item["train_class_counts"].items()},
                validation_class_counts={int(k): int(v) for k, v in item["validation_class_counts"].items()},
                seed=int(item["seed"]),
                split_hash=item["split_hash"],
            )
            for item in outer["inner_folds"]
        )
        outer_defs.append(
            OuterFoldDefinition(
                outer_fold_id=outer["outer_fold_id"],
                repeat_index=int(outer["repeat_index"]),
                fold_index=int(outer["fold_index"]),
                train_indices=tuple(outer["train_indices"]),
                test_indices=tuple(outer["test_indices"]),
                train_class_counts={int(k): int(v) for k, v in outer["train_class_counts"].items()},
                test_class_counts={int(k): int(v) for k, v in outer["test_class_counts"].items()},
                seed=int(outer["seed"]),
                split_hash=outer["split_hash"],
                inner_folds=inner_defs,
            )
        )
    definition = NestedCVDefinition(
        dataset_id=folds["dataset_id"],
        row_count=int(folds["row_count"]),
        outer_strategy=folds["outer_strategy"],
        outer_n_repeats=int(folds["outer_n_repeats"]),
        outer_n_splits=int(folds["outer_n_splits"]),
        inner_strategy=folds["inner_strategy"],
        inner_n_splits=int(folds["inner_n_splits"]),
        shuffle=bool(folds["shuffle"]),
        base_seed=int(folds["base_seed"]),
        dataset_checksum=folds["dataset_checksum"],
        outer_folds=tuple(outer_defs),
        nested_cv_hash=folds["nested_cv_hash"],
    )
    try:
        validate_nested_cv_definition(definition, target)
    except SplitError as exc:
        raise ArtifactError(str(exc)) from exc
