"""P3C nested CV artifact writer and validator."""

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
from creditrep.config.nested import NestedCVConfig
from creditrep.datasets.models import LoadedDataset
from creditrep.datasets.registry import find_repo_root, resolve_repo_path
from creditrep.experiments.nested_cv import NestedCVValidationResult
from creditrep.preprocessing.protocol import ProtocolAConfig
from creditrep.splitting.hashing import split_hash
from creditrep.splitting.exceptions import SplitError
from creditrep.splitting.nested import NestedCVDefinition, validate_nested_cv_definition

NESTED_ARTIFACT_SCHEMA_VERSION = "0.1.0"


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


def build_nested_experiment_id(
    config: NestedCVConfig,
    nested_cv_hash: str,
    *,
    created_at_utc: datetime | None = None,
) -> str:
    timestamp = (created_at_utc or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{_slugify(config.experiment_name)}-{timestamp}-{nested_cv_hash[:8]}"


def protocol_config_hash(protocol_config: ProtocolAConfig) -> str:
    return split_hash({"protocol_config": protocol_config.__dict__})


def nested_config_hash(config: NestedCVConfig) -> str:
    return split_hash(config.canonical_payload())


def create_nested_cv_artifact(
    *,
    config: NestedCVConfig,
    protocol_config: ProtocolAConfig,
    dataset: LoadedDataset,
    checksum: DatasetChecksum,
    result: NestedCVValidationResult,
    repo_root: Path | str | None = None,
    created_at_utc: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write non-publishable P3C nested CV artifact atomically."""

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    output_root = resolve_repo_path(config.output_root, repo_root=root, context="output.root_dir")
    created = created_at_utc or datetime.now(timezone.utc)
    experiment_id = build_nested_experiment_id(config, result.nested_cv.nested_cv_hash, created_at_utc=created)
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
                "config_hash": nested_config_hash(config),
                "created_at_utc": created.isoformat().replace("+00:00", "Z"),
                "package_version": PACKAGE_VERSION,
                "protocol_config_hash": protocol_config_hash(protocol_config),
            }
        )
        manifest = {
            "schema_version": NESTED_ARTIFACT_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "experiment_name": config.experiment_name,
            "status": "completed",
            "result_scope": "preprocessing_validation",
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
            "provenance": provenance,
            "reserved_artifacts": {"metrics": None, "predictions": None, "trained_model": None, "plots": None},
        }
        _write_json(temp_dir / "manifest.json", manifest)
        with (temp_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.canonical_payload(), handle, sort_keys=True, allow_unicode=True)
        _write_json(nested_dir / "manifest.json", manifest["nested_cv"] | {"schema_version": NESTED_ARTIFACT_SCHEMA_VERSION})
        _write_json(nested_dir / "outer_folds.json", result.nested_cv.to_dict())
        _write_outer_csv(nested_dir / "outer_folds.csv", result.nested_cv)
        for outer in result.nested_cv.outer_folds:
            outer_path = outer_dir / outer.outer_fold_id
            inner_path = outer_path / "inner"
            inner_path.mkdir(parents=True)
            _write_json(outer_path / "split.json", outer.to_dict())
            _write_json(outer_path / "preprocessing.json", result.outer_preprocessing[outer.outer_fold_id])
            _write_json(outer_path / "tuning_summary.json", result.tuning_summaries[outer.outer_fold_id])
            for inner in outer.inner_folds:
                fold_path = inner_path / inner.inner_fold_id
                fold_path.mkdir()
                _write_json(fold_path / "split.json", inner.to_dict())
                _write_json(fold_path / "preprocessing.json", result.inner_preprocessing[inner.inner_fold_id])
        validate_nested_cv_artifact(temp_dir, target=dataset.target)
        temp_dir.rename(final_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return final_dir, manifest


def load_nested_cv_artifact(artifact_dir: Path | str) -> dict[str, Any]:
    path = Path(artifact_dir)
    required = [path / "manifest.json", path / "config.yaml", path / "nested_cv" / "outer_folds.json"]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise ArtifactError(f"Nested CV artifact missing required files: {missing}.")
    with (path / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (path / "nested_cv" / "outer_folds.json").open(encoding="utf-8") as handle:
        folds = json.load(handle)
    return {"manifest": manifest, "outer_folds": folds}


def validate_nested_cv_artifact(artifact_dir: Path | str, *, target) -> None:
    """Validate schema, flags and nested CV hash for a written artifact."""

    loaded = load_nested_cv_artifact(artifact_dir)
    manifest = loaded["manifest"]
    folds = loaded["outer_folds"]
    if manifest.get("publishable") is not False:
        raise ArtifactError("Nested CV artifact must set publishable=false.")
    if manifest.get("result_scope") != "preprocessing_validation":
        raise ArtifactError("Nested CV artifact result_scope must be preprocessing_validation.")
    expected_hash = split_hash({key: value for key, value in folds.items() if key != "nested_cv_hash"})
    if expected_hash != folds.get("nested_cv_hash"):
        raise ArtifactError("Nested CV hash mismatch.")
    if manifest["nested_cv"]["nested_cv_hash"] != folds["nested_cv_hash"]:
        raise ArtifactError("Manifest nested_cv_hash mismatch.")
    # Rebuild dataclasses lightly enough to reuse structural validator.
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
