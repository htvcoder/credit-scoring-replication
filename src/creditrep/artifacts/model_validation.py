"""Portable, atomic artifacts for non-publishable P5C model validation."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd
import yaml

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.neural import (
    NEURAL_ARTIFACT_SCHEMA_VERSION,
    RESULT_SCOPE as NEURAL_RESULT_SCOPE,
    validate_neural_fold_artifact_set,
)
from creditrep.artifacts.git import get_git_provenance
from creditrep.config.loader import sha256_canonical
from creditrep.config.model_validation import ModelValidationConfig

if TYPE_CHECKING:
    from creditrep.experiments.model_validation import ModelValidationResult

SCHEMA_VERSION = "1.1"
PREDICTION_COLUMNS = [
    "row_position",
    "outer_repeat",
    "outer_fold",
    "partition",
    "y_true",
    "y_score",
    "y_pred",
]
FAILURE_STAGES = {
    "loading",
    "fold_validation",
    "preprocessing",
    "inner_tuning",
    "outer_refit",
    "prediction",
    "metrics",
    "artifact_write",
    "artifact_validation",
    "summary_update",
    "early_stopping_split",
    "neural_model_initialization",
    "neural_training",
    "neural_metadata_capture",
    "neural_artifact_validation",
    "neural_artifact_publication",
    "neural_reconciliation",
}
NEURAL_FAILURE_STAGES = FAILURE_STAGES - {
    "loading",
    "fold_validation",
    "preprocessing",
    "inner_tuning",
    "outer_refit",
    "prediction",
    "metrics",
    "artifact_write",
    "artifact_validation",
    "summary_update",
}
RETRY_CLASSES = {"retryable", "non_retryable"}


class ArtifactValidationError(ArtifactError):
    """A complete temporary fold failed validation and was not promoted."""


def execution_unit_id(
    *,
    experiment_id: str,
    dataset_checksum: str | None,
    config_fingerprint: str,
    model_id: str,
    outer_fold_id: str,
    inner_fold_id: str | None = None,
    candidate_id: str | int | None = None,
    training_scope: str | None = None,
) -> str:
    """Stable identity for a retryable execution boundary; never includes runtime."""
    return sha256_canonical(
        {
            "experiment_id": experiment_id,
            "dataset_checksum": dataset_checksum,
            "config_fingerprint": config_fingerprint,
            "model_id": model_id,
            "outer_fold_id": outer_fold_id,
            "inner_fold_id": inner_fold_id,
            "candidate_id": candidate_id,
            "training_scope": training_scope,
        }
    )


def classify_retry(exception: Exception, *, stage: str) -> tuple[str, bool]:
    """Conservative deterministic retry policy: data/config defects are never retried."""
    text = str(exception).lower()
    if stage in {
        "early_stopping_split",
        "neural_metadata_capture",
        "neural_artifact_validation",
    }:
        return "non_retryable", False
    if stage == "neural_reconciliation":
        if any(
            marker in text
            for marker in ("malformed", "incomplete", "missing", "corrupt")
        ):
            return "retryable", True
        return "non_retryable", False
    if any(
        marker in text
        for marker in (
            "non-finite",
            "nan",
            "infinity",
            "unsupported",
            "mismatch",
            "invalid",
            "schema",
            "probability",
        )
    ):
        return "non_retryable", False
    if isinstance(exception, (OSError, TimeoutError)) or any(
        marker in text for marker in ("lock", "interrupted", "temporar", "disk error")
    ):
        return "retryable", True
    return "non_retryable", False


def _sanitize_failure_message(exception: Exception) -> str:
    message = str(exception).replace("\\", "/").splitlines()[0]
    message = re.sub(
        r"(?i)\b(secret|token|password)\s*=\s*[^\s,;]+", r"\1=[REDACTED]", message
    )
    message = re.sub(r"(?i)(?:[a-z]:)?/(?:[^\s/]+/)+[^\s/]+", "[PATH]", message)
    message = re.sub(r"(?i)raw[-_ ]?row\s*=\s*[^\s,;]+", "raw-row=[REDACTED]", message)
    message = re.sub(r"(?i)tensor\([^)]*\)", "Tensor([REDACTED])", message)
    message = re.sub(r"(?i)state_dict\s*=\s*[^\s,;]+", "state_dict=[REDACTED]", message)
    return message[:500]


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Malformed {label}: {path.name}.") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Malformed {label}: {path.name} must contain an object.")
    return value


def _require_non_publishable(value: dict[str, Any], label: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"{label} schema version mismatch.")
    if (
        value.get("publishable") is not False
        or value.get("result_scope") != "model_validation"
    ):
        raise ArtifactError(
            f"{label} is not a non-publishable model_validation artifact."
        )


def validate_summary_artifact(
    path: Path | str,
    *,
    config_hash: str | None = None,
    dataset_checksum: str | None = None,
) -> dict[str, Any]:
    summary = _load_json(Path(path), "summary")
    _require_non_publishable(summary, "Summary")
    required = {
        "planned_fold_count",
        "completed_fold_count",
        "failed_fold_count",
        "config_hash",
    }
    if required - set(summary):
        raise ArtifactError("Summary has an invalid schema.")
    count_keys = ("planned_fold_count", "completed_fold_count", "failed_fold_count")
    if any(
        not isinstance(summary[key], int)
        or isinstance(summary[key], bool)
        or summary[key] < 0
        for key in count_keys
    ):
        raise ArtifactError("Summary counts must be non-negative integers.")
    pending = summary.get("pending_fold_count", 0)
    invalid = summary.get("invalid_fold_count", 0)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (pending, invalid)
    ):
        raise ArtifactError("Summary counts must be non-negative integers.")
    if (
        summary["completed_fold_count"]
        + summary["failed_fold_count"]
        + pending
        + invalid
        != summary["planned_fold_count"]
    ):
        raise ArtifactError("Summary counts do not reconcile.")
    if config_hash is not None and summary["config_hash"] != config_hash:
        raise ArtifactError("Summary config hash mismatch.")
    if dataset_checksum is not None and summary.get("dataset_checksum") not in (
        None,
        dataset_checksum,
    ):
        raise ArtifactError("Summary dataset checksum mismatch.")
    for key in (
        "planned_units",
        "completed_units",
        "failed_units",
        "invalid_units",
        "pending_units",
    ):
        if key in summary and (
            not isinstance(summary[key], list)
            or summary[key] != sorted(set(summary[key]))
        ):
            raise ArtifactError(f"Summary {key} must be a sorted unique list.")
    return summary


def validate_failure_artifact(
    path: Path | str, *, config_hash: str | None = None
) -> dict[str, Any]:
    failure = _load_json(Path(path), "failure artifact")
    _require_non_publishable(failure, "Failure artifact")
    required = {
        "experiment_id",
        "dataset_id",
        "model_id",
        "fold_id",
        "fold_hash",
        "stage",
        "exception_type",
        "message",
        "retryable",
        "config_hash",
        "timestamp_utc",
        "cleanup_status",
        "attempt",
        "first_failure_timestamp_utc",
        "resolved",
    }
    if (
        required - set(failure)
        or failure["stage"] not in FAILURE_STAGES
        or not isinstance(failure["message"], str)
        or len(failure["message"]) > 500
        or not isinstance(failure["attempt"], int)
        or isinstance(failure["attempt"], bool)
        or failure["attempt"] < 1
        or not isinstance(failure["resolved"], bool)
        or not isinstance(failure["retryable"], bool)
        or not all(
            isinstance(failure.get(key), str) and failure[key]
            for key in (
                "experiment_id",
                "dataset_id",
                "model_id",
                "fold_id",
                "fold_hash",
                "config_hash",
            )
        )
    ):
        raise ArtifactError("Failure artifact has an invalid schema.")
    if config_hash is not None and failure["config_hash"] != config_hash:
        raise ArtifactError("Failure artifact config hash mismatch.")
    if failure["stage"] in NEURAL_FAILURE_STAGES:
        neural = {
            "outer_fold_id",
            "training_scope",
            "execution_unit_id",
            "retry_class",
            "attempt_number",
            "config_fingerprint",
            "model_seed",
        }
        if (
            neural - set(failure)
            or failure["retry_class"] not in RETRY_CLASSES
            or failure["retryable"] != (failure["retry_class"] == "retryable")
        ):
            raise ArtifactError(
                "Neural failure artifact has an invalid retry/context schema."
            )
        if failure["training_scope"] not in {"inner_candidate", "final_refit"}:
            raise ArtifactError("Neural failure artifact has invalid training_scope.")
        if failure["training_scope"] == "inner_candidate" and (
            failure.get("inner_fold_id") is None or failure.get("candidate_id") is None
        ):
            raise ArtifactError(
                "Neural inner-candidate failure lacks identity context."
            )
    return failure


def write_failure_artifact(
    *,
    root: Path | str,
    experiment_id: str,
    dataset_id: str,
    model_id: str,
    fold_id: str,
    fold_hash: str,
    stage: str,
    exception: Exception,
    config_hash: str,
    retryable: bool = True,
    cleanup_status: str = "completed",
    neural_context: dict[str, Any] | None = None,
) -> Path:
    if stage not in FAILURE_STAGES:
        raise ArtifactError(f"Unsupported failure stage: {stage}.")
    # Keep user-controlled exception output bounded and avoid leaking paths/tracebacks.
    message = _sanitize_failure_message(exception)
    target = Path(root) / "failures" / f"{fold_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    old = None
    if target.exists():
        try:
            old = validate_failure_artifact(target, config_hash=config_hash)
        except ArtifactError:
            old = None
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    retry_class, classified_retryable = classify_retry(exception, stage=stage)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "fold_id": fold_id,
        "fold_hash": fold_hash,
        "stage": stage,
        "exception_type": type(exception).__name__,
        "message": message,
        "retryable": bool(
            retryable if neural_context is None else retryable and classified_retryable
        ),
        "config_hash": config_hash,
        "timestamp_utc": now,
        "first_failure_timestamp_utc": old.get("first_failure_timestamp_utc", now)
        if old
        else now,
        "attempt": int(old.get("attempt", 0)) + 1 if old else 1,
        "resolved": False,
        "cleanup_status": cleanup_status,
        "publishable": False,
        "result_scope": "model_validation",
    }
    if neural_context is not None:
        payload |= dict(neural_context) | {
            "retry_class": retry_class,
            "retryable": classified_retryable,
            "attempt_number": int(old.get("attempt_number", old.get("attempt", 0))) + 1
            if old
            else 1,
        }
    temp = target.with_suffix(".tmp")
    _json(temp, payload)
    validate_failure_artifact(temp, config_hash=config_hash)
    temp.replace(target)
    return target


def resolve_failure_artifact(root: Path | str, fold_id: str) -> None:
    """Retain failure evidence while marking it resolved after a successful retry."""
    target = Path(root) / "failures" / f"{fold_id}.json"
    if not target.exists():
        return
    failure = _load_json(target, "failure artifact")
    failure["resolved"] = True
    failure["resolved_timestamp_utc"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    temp = target.with_suffix(".tmp")
    _json(temp, failure)
    temp.replace(target)


def validate_fold(
    path: Path,
    *,
    config_hash: str | None = None,
    dataset_checksum: str | None = None,
    fold_hash: str | None = None,
    model_id: str | None = None,
    experiment_id: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    required = [
        "fold_metadata.json",
        "tuning.json",
        "preprocessing.json",
        "model_metadata.json",
        "metrics.json",
        "predictions.csv",
        "complete.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ArtifactError(
            f"Incomplete model-validation fold {path.name}: missing {missing}."
        )
    metadata = _load_json(path / "fold_metadata.json", "fold metadata")
    _require_non_publishable(metadata, "Fold metadata")
    required_metadata = {
        "experiment_id",
        "dataset_id",
        "fold_id",
        "fold_hash",
        "model_id",
        "outer_repeat",
        "outer_fold",
        "dataset_checksum",
        "config_hash",
        "test_count",
    }
    if required_metadata - set(metadata):
        raise ArtifactError(
            f"Fold metadata has an invalid schema for fold {path.name}."
        )
    if path.name not in {metadata["fold_id"], f".tmp-{metadata['fold_id']}"}:
        raise ArtifactError(f"Fold ID mismatch for fold {path.name}.")
    if experiment_id is not None and metadata["experiment_id"] != experiment_id:
        raise ArtifactError(f"Experiment ID mismatch for fold {path.name}.")
    if dataset_id is not None and metadata["dataset_id"] != dataset_id:
        raise ArtifactError(f"Dataset ID mismatch for fold {path.name}.")
    complete = _load_json(path / "complete.json", "complete marker")
    if complete.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Complete marker schema version mismatch for fold {path.name}."
        )
    if complete.get("config_hash") != metadata.get("config_hash"):
        raise ArtifactError(
            f"Complete marker config hash mismatch for fold {path.name}."
        )
    if complete.get("fold_hash") != metadata.get("fold_hash"):
        raise ArtifactError(f"Complete marker fold hash mismatch for fold {path.name}.")
    for name in (
        "tuning.json",
        "preprocessing.json",
        "model_metadata.json",
        "metrics.json",
    ):
        value = _load_json(path / name, name)
        if name == "metrics.json":
            _require_non_publishable(value, "Metrics")
    if config_hash is not None and metadata.get("config_hash") != config_hash:
        raise ArtifactError(f"Config hash mismatch for fold {path.name}.")
    if (
        dataset_checksum is not None
        and metadata.get("dataset_checksum") != dataset_checksum
    ):
        raise ArtifactError(f"Dataset checksum mismatch for fold {path.name}.")
    if fold_hash is not None and metadata.get("fold_hash") != fold_hash:
        raise ArtifactError(f"Fold hash mismatch for fold {path.name}.")
    model = _load_json(path / "model_metadata.json", "model metadata")
    if model.get("model_id") != metadata.get("model_id") or (
        model_id is not None and model["model_id"] != model_id
    ):
        raise ArtifactError(f"Model ID mismatch for fold {path.name}.")
    if model.get("model_id") == "decision_tree" and (
        model.get("algorithm"),
        model.get("implementation"),
        model.get("replication_role"),
        model.get("deviation_from_paper"),
    ) != (
        "cart",
        "sklearn.tree.DecisionTreeClassifier",
        "approximation",
        "c45_to_cart",
    ):
        raise ArtifactError("Decision Tree CART provenance is missing.")
    try:
        predictions = pd.read_csv(path / "predictions.csv")
    except Exception as exc:
        raise ArtifactError(f"Malformed predictions for fold {path.name}.") from exc
    if (
        list(predictions.columns) != PREDICTION_COLUMNS
        or predictions.empty
        or predictions["row_position"].duplicated().any()
        or set(predictions["partition"]) != {"test"}
        or predictions.isna().any().any()
    ):
        raise ArtifactError(f"Invalid predictions for fold {path.name}.")
    for column in ("row_position", "outer_repeat", "outer_fold", "y_true", "y_pred"):
        numeric = pd.to_numeric(predictions[column], errors="coerce")
        if (
            numeric.isna().any()
            or not numeric.map(lambda value: float(value).is_integer()).all()
        ):
            raise ArtifactError(f"Invalid {column} values for fold {path.name}.")
        predictions[column] = numeric.astype("int64")
    if (predictions["row_position"] < 0).any():
        raise ArtifactError(f"Invalid row_position values for fold {path.name}.")
    if not predictions["y_true"].isin([0, 1]).all():
        raise ArtifactError(f"Invalid y_true values for fold {path.name}.")
    if not predictions["y_pred"].isin([0, 1]).all():
        raise ArtifactError(f"Invalid y_pred values for fold {path.name}.")
    if set(predictions["outer_repeat"]) != {metadata["outer_repeat"]} or set(
        predictions["outer_fold"]
    ) != {metadata["outer_fold"]}:
        raise ArtifactError(
            f"Prediction fold provenance mismatch for fold {path.name}."
        )
    try:
        scores = pd.to_numeric(predictions["y_score"], errors="raise")
    except Exception as exc:
        raise ArtifactError(
            f"Invalid prediction probabilities for fold {path.name}."
        ) from exc
    if (
        not scores.map(lambda value: float("-inf") < float(value) < float("inf")).all()
        or not scores.between(0, 1).all()
    ):
        raise ArtifactError(f"Invalid prediction probabilities for fold {path.name}.")
    if len(predictions) != metadata.get("test_count"):
        raise ArtifactError(f"Prediction row-count mismatch for fold {path.name}.")
    validate_neural_fold_artifact_set(path, expected_model_id=metadata["model_id"])
    return metadata


def _fold_payload(
    fold_id: str,
    fold: dict[str, Any],
    config: ModelValidationConfig,
    dataset_checksum: str | None,
    experiment_id: str,
) -> dict[str, Any]:
    outer_id = fold["outer_fold_id"]
    repeat, index = 0, 0
    if outer_id.startswith("repeat_"):
        parts = outer_id.split("_")
        repeat, index = int(parts[1]), int(parts[3])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset_id": config.dataset_id,
        "fold_id": fold_id,
        "fold_hash": fold["fold_hash"],
        "model_id": fold["model_id"],
        "outer_fold_id": outer_id,
        "outer_repeat": repeat,
        "outer_fold": index,
        "dataset_checksum": dataset_checksum,
        "config_hash": config.config_hash,
        "train_count": fold.get("train_count"),
        "test_count": len(fold["predictions"]),
        "train_class_counts": fold.get("train_class_counts", {}),
        "test_class_counts": fold.get("test_class_counts", {}),
        "inner_fold_count": fold.get("inner_fold_count"),
        "timings": fold["timings"],
        "warnings": fold.get("warnings", []),
        "publishable": False,
        "result_scope": "model_validation",
    }


def initialise_experiment(
    *,
    root: Path | str,
    config: ModelValidationConfig,
    experiment_id: str,
    dataset_checksum: str | None,
    planned_fold_count: int,
    repo_root: Path | str | None = None,
) -> Path:
    """Create only experiment metadata; fold directories are promoted independently."""
    final = Path(root) / experiment_id
    final.mkdir(parents=True, exist_ok=True)
    manifest_path = final / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path, "manifest")
        _require_non_publishable(manifest, "Manifest")
        if (
            manifest.get("experiment_id") != experiment_id
            or manifest.get("config_hash") != config.config_hash
            or manifest.get("dataset", {}).get("id") != config.dataset_id
            or manifest.get("dataset", {}).get("checksum_sha256") != dataset_checksum
        ):
            raise ArtifactError(
                "Existing experiment provenance mismatch; use a new experiment ID."
            )
        return final
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    provenance = get_git_provenance(
        Path(repo_root) if repo_root is not None else Path.cwd()
    ).to_dict()
    _json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "experiment_name": config.experiment_name,
            "result_scope": "model_validation",
            "publishable": False,
            "dataset": {"id": config.dataset_id, "checksum_sha256": dataset_checksum},
            "config_hash": config.config_hash,
            "provenance": provenance,
            "artifact_state": "running",
            "planned_fold_count": planned_fold_count,
            "created_at_utc": now,
            "updated_at_utc": now,
        },
    )
    (final / "config.yaml").write_text(
        yaml.safe_dump(config.canonical_payload(), sort_keys=True), encoding="utf-8"
    )
    (final / "folds").mkdir(exist_ok=True)
    return final


def write_fold_artifact(
    *,
    experiment_root: Path | str,
    config: ModelValidationConfig,
    dataset_checksum: str | None,
    fold_id: str,
    fold: dict[str, Any],
) -> Path:
    root = Path(experiment_root)
    destination = root / "folds" / fold_id
    if destination.exists():
        raise ArtifactError(f"Refusing to overwrite existing fold artifact: {fold_id}.")
    temporary = root / "folds" / f".tmp-{fold_id}"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        temporary.mkdir(parents=True)
        metadata = _fold_payload(fold_id, fold, config, dataset_checksum, root.name)
        _json(temporary / "fold_metadata.json", metadata)
        _json(
            temporary / "tuning.json",
            {
                "selected_candidate": fold["selected_candidate"],
                "candidate_hashes": [fold["selected_candidate"]["candidate_hash"]],
                "inner_scores": fold["selected_candidate"]["inner_scores"],
            },
        )
        _json(temporary / "preprocessing.json", fold["preprocessing"])
        _json(temporary / "model_metadata.json", fold["model_metadata"])
        _json(
            temporary / "metrics.json",
            {
                "schema_version": SCHEMA_VERSION,
                "metrics": fold["metrics"],
                "publishable": False,
                "result_scope": "model_validation",
            },
        )
        predictions = fold["predictions"].copy()
        predictions["outer_repeat"] = metadata["outer_repeat"]
        predictions["outer_fold"] = metadata["outer_fold"]
        predictions["partition"] = "test"
        predictions[PREDICTION_COLUMNS].to_csv(
            temporary / "predictions.csv", index=False
        )
        neural = fold.get("neural_artifacts")
        if neural is not None:
            neural_root = temporary / "neural"
            manifest_candidates = []
            for candidate in neural["candidates"]:
                candidate_id = candidate["candidate_id"]
                candidate_root = (
                    neural_root / "candidates" / f"candidate-{candidate_id}"
                )
                runs = []
                for evidence in candidate["runs"]:
                    inner_id = evidence["training_summary"]["inner_fold_id"]
                    run_root = candidate_root / f"inner-{inner_id}"
                    run_root.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for name, payload in evidence.items():
                        filename = f"{name}.json"
                        _json(run_root / filename, payload)
                        paths[name] = str(
                            (run_root / filename).relative_to(temporary)
                        ).replace("\\", "/")
                    runs.append(paths)
                manifest_candidates.append(
                    {
                        key: candidate[key]
                        for key in (
                            "candidate_id",
                            "candidate_hash",
                            "inner_scores",
                            "selection_metric",
                            "selected",
                        )
                    }
                    | {"runs": runs}
                )
            final_paths = {"selected_candidate_id": neural["selected_candidate_id"]}
            final_root = neural_root / "final_refit"
            final_root.mkdir(parents=True, exist_ok=True)
            for name, payload in neural["final_refit"].items():
                filename = f"{name}.json"
                _json(final_root / filename, payload)
                final_paths[name] = str(
                    (final_root / filename).relative_to(temporary)
                ).replace("\\", "/")
            _json(
                neural_root / "neural_manifest.json",
                {
                    "schema_version": NEURAL_ARTIFACT_SCHEMA_VERSION,
                    "experiment_id": root.name,
                    "dataset_id": config.dataset_id,
                    "model_id": fold["model_id"],
                    "outer_fold_id": fold["outer_fold_id"],
                    "publishable": False,
                    "result_scope": NEURAL_RESULT_SCOPE,
                    "fair_budget_id": neural["fair_budget_id"],
                    "selected_candidate_id": neural["selected_candidate_id"],
                    "candidates": manifest_candidates,
                    "final_refit": final_paths,
                },
            )
        _json(
            temporary / "complete.json",
            {
                "schema_version": SCHEMA_VERSION,
                "fold_hash": fold["fold_hash"],
                "config_hash": config.config_hash,
            },
        )
        try:
            validate_fold(
                temporary,
                config_hash=config.config_hash,
                dataset_checksum=dataset_checksum,
                fold_hash=fold["fold_hash"],
                model_id=fold["model_id"],
                experiment_id=root.name,
                dataset_id=config.dataset_id,
            )
        except ArtifactError as exc:
            raise ArtifactValidationError(str(exc)) from exc
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def reconcile_fold_state(
    *,
    root: Path | str,
    fold_id: str,
    config: ModelValidationConfig,
    dataset_checksum: str | None,
    fold_hash: str,
    model_id: str,
) -> str:
    """Return a deterministic state without trusting directory enumeration order."""
    experiment = Path(root)
    fold = experiment / "folds" / fold_id
    failure = experiment / "failures" / f"{fold_id}.json"
    if fold.exists():
        try:
            validate_fold(
                fold,
                config_hash=config.config_hash,
                dataset_checksum=dataset_checksum,
                fold_hash=fold_hash,
                model_id=model_id,
                experiment_id=experiment.name,
                dataset_id=config.dataset_id,
            )
            return "valid_completed"
        except ArtifactError as exc:
            message = str(exc).lower()
            if "schema version" in message:
                return "unsupported_schema"
            if "config hash mismatch" in message or "experiment id mismatch" in message:
                return "identity_mismatch"
            return "corrupt"
    if failure.exists():
        try:
            payload = validate_failure_artifact(failure, config_hash=config.config_hash)
        except ArtifactError as exc:
            return "unsupported_schema" if "schema" in str(exc).lower() else "corrupt"
        return "retryable_failed" if payload["retryable"] else "non_retryable_failed"
    if (experiment / "folds" / f".tmp-{fold_id}").exists():
        return "incomplete"
    return "incomplete"


def reconcile_summary(
    *,
    experiment_root: Path | str,
    planned_units: dict[str, dict[str, str]],
    config: ModelValidationConfig,
    dataset_checksum: str | None,
) -> dict[str, Any]:
    """Rebuild summary from fold/failure artifacts rather than trusting an old summary."""
    root = Path(experiment_root)
    completed_units: list[str] = []
    failed_units: list[str] = []
    invalid_units: list[str] = []
    total_attempts = 0
    for fold_id, expected in sorted(planned_units.items()):
        fold_dir = root / "folds" / fold_id
        failure_path = root / "failures" / f"{fold_id}.json"
        if fold_dir.exists():
            try:
                validate_fold(
                    fold_dir,
                    config_hash=config.config_hash,
                    dataset_checksum=dataset_checksum,
                    fold_hash=expected["fold_hash"],
                    model_id=expected["model_id"],
                )
                completed_units.append(fold_id)
                total_attempts += 1
            except ArtifactError:
                invalid_units.append(fold_id)
        if failure_path.exists():
            try:
                failure = validate_failure_artifact(
                    failure_path, config_hash=config.config_hash
                )
                total_attempts += int(failure["attempt"])
                if not failure.get("resolved"):
                    failed_units.append(fold_id)
            except ArtifactError:
                invalid_units.append(fold_id)
    completed_units = sorted(set(completed_units))
    failed_units = sorted(set(failed_units) - set(completed_units))
    invalid_units = sorted(
        set(invalid_units) - set(completed_units) - set(failed_units)
    )
    pending_units = sorted(
        set(planned_units)
        - set(completed_units)
        - set(failed_units)
        - set(invalid_units)
    )
    completed, failed, invalid, pending = map(
        len, (completed_units, failed_units, invalid_units, pending_units)
    )
    state = (
        "completed"
        if completed == len(planned_units)
        else (
            "completed_with_failures"
            if completed and failed
            else "failed"
            if failed and not completed
            else "partial"
        )
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "planned_units": sorted(planned_units),
        "completed_units": completed_units,
        "failed_units": failed_units,
        "invalid_units": invalid_units,
        "pending_units": pending_units,
        "planned_fold_count": len(planned_units),
        "completed_fold_count": completed,
        "failed_fold_count": failed,
        "pending_fold_count": pending,
        "invalid_fold_count": invalid,
        "resumed_skipped_fold_count": 0,
        "retried_fold_count": 0,
        "total_attempts": total_attempts,
        "overall_state": state,
        "publishable": False,
        "result_scope": "model_validation",
        "config_hash": config.config_hash,
        "dataset_checksum": dataset_checksum,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    temp = root / "summary.tmp"
    _json(temp, summary)
    temp.replace(root / "summary.json")
    manifest = _load_json(root / "manifest.json", "manifest")
    manifest.update(
        {
            "artifact_state": state,
            "completed_fold_count": completed,
            "failed_fold_count": failed,
            "updated_at_utc": summary["updated_at_utc"],
        }
    )
    _json(root / "manifest.tmp", manifest)
    (root / "manifest.tmp").replace(root / "manifest.json")
    return summary


def write_model_validation_artifact(
    *,
    config: ModelValidationConfig,
    result: ModelValidationResult,
    output_root: Path | str,
    experiment_id: str | None = None,
    dataset_checksum: str | None = None,
    repo_root: Path | str | None = None,
    resume: bool = False,
) -> Path:
    root = Path(output_root)
    experiment_id = (
        experiment_id or f"{config.experiment_name}-{config.config_hash[:12]}"
    )
    final = root / experiment_id
    if final.exists():
        if resume:
            manifest = validate_model_validation_artifact(
                final, config_hash=config.config_hash, dataset_checksum=dataset_checksum
            )
            return final
        raise ArtifactError(
            f"Artifact already exists: {final}. Use resume only with a validated completed artifact."
        )
    temporary = root / f".tmp-{experiment_id}"
    root.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        temporary.mkdir()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        provenance = get_git_provenance(
            Path(repo_root) if repo_root is not None else Path.cwd()
        ).to_dict()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "experiment_name": config.experiment_name,
            "result_scope": "model_validation",
            "publishable": False,
            "dataset": {"id": config.dataset_id, "checksum_sha256": dataset_checksum},
            "config_hash": config.config_hash,
            "provenance": provenance,
            "artifact_state": "completed",
            "planned_fold_count": result.summary["planned_fold_count"],
            "completed_fold_count": result.summary["completed_fold_count"],
            "failed_fold_count": result.summary["failed_fold_count"],
            "skipped_resumed_fold_count": 0,
            "created_at_utc": now,
            "updated_at_utc": now,
        }
        _json(temporary / "manifest.json", manifest)
        (temporary / "config.yaml").write_text(
            yaml.safe_dump(config.canonical_payload(), sort_keys=True), encoding="utf-8"
        )
        folds_dir = temporary / "folds"
        folds_dir.mkdir()
        for fold_id, fold in result.folds.items():
            target = folds_dir / fold_id
            target.mkdir()
            metadata = _fold_payload(
                fold_id, fold, config, dataset_checksum, experiment_id
            )
            _json(target / "fold_metadata.json", metadata)
            _json(
                target / "tuning.json",
                {
                    "selected_candidate": fold["selected_candidate"],
                    "candidate_hashes": [fold["selected_candidate"]["candidate_hash"]],
                    "inner_scores": fold["selected_candidate"]["inner_scores"],
                },
            )
            _json(target / "preprocessing.json", fold["preprocessing"])
            _json(target / "model_metadata.json", fold["model_metadata"])
            _json(
                target / "metrics.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "metrics": fold["metrics"],
                    "publishable": False,
                    "result_scope": "model_validation",
                },
            )
            predictions = fold["predictions"].copy()
            predictions["outer_repeat"] = metadata["outer_repeat"]
            predictions["outer_fold"] = metadata["outer_fold"]
            predictions["partition"] = "test"
            predictions = predictions[PREDICTION_COLUMNS]
            predictions.to_csv(target / "predictions.csv", index=False)
            _json(
                target / "complete.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "fold_hash": fold["fold_hash"],
                    "config_hash": config.config_hash,
                },
            )
            validate_fold(
                target,
                config_hash=config.config_hash,
                dataset_checksum=dataset_checksum,
            )
        _json(
            temporary / "summary.json",
            result.summary | {"schema_version": SCHEMA_VERSION},
        )
        validate_model_validation_artifact(
            temporary, config_hash=config.config_hash, dataset_checksum=dataset_checksum
        )
        temporary.rename(final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final


def validate_model_validation_artifact(
    path: Path | str,
    *,
    config_hash: str | None = None,
    dataset_checksum: str | None = None,
) -> dict[str, Any]:
    root = Path(path)
    manifest = _load_json(root / "manifest.json", "manifest")
    _require_non_publishable(manifest, "Manifest")
    required_manifest = {
        "experiment_id",
        "experiment_name",
        "dataset",
        "config_hash",
        "artifact_state",
        "planned_fold_count",
        "completed_fold_count",
        "failed_fold_count",
    }
    if required_manifest - set(manifest) or not isinstance(
        manifest.get("dataset"), dict
    ):
        raise ArtifactError("Manifest has an invalid schema.")
    if manifest.get("artifact_state") != "completed":
        raise ArtifactError("Model-validation artifact is not complete.")
    if config_hash is not None and manifest.get("config_hash") != config_hash:
        raise ArtifactError("Manifest config hash mismatch.")
    actual_checksum = manifest.get("dataset", {}).get("checksum_sha256")
    if dataset_checksum is not None and actual_checksum != dataset_checksum:
        raise ArtifactError("Manifest dataset checksum mismatch.")
    folds_dir = root / "folds"
    if (
        not folds_dir.is_dir()
        or not (root / "config.yaml").is_file()
        or not (root / "summary.json").is_file()
    ):
        raise ArtifactError("Incomplete model-validation artifact set.")
    try:
        config_payload = yaml.safe_load(
            (root / "config.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError("Malformed config artifact.") from exc
    if (
        not isinstance(config_payload, dict)
        or sha256_canonical(config_payload) != manifest["config_hash"]
    ):
        raise ArtifactError("Manifest/config cross-file mismatch.")
    summary = validate_summary_artifact(
        root / "summary.json",
        config_hash=manifest["config_hash"],
        dataset_checksum=actual_checksum,
    )
    folds = sorted(item for item in folds_dir.iterdir() if item.is_dir())
    for fold in folds:
        validate_fold(
            fold,
            config_hash=manifest["config_hash"],
            dataset_checksum=actual_checksum,
            experiment_id=manifest["experiment_id"],
            dataset_id=manifest["dataset"]["id"],
        )
    if len(folds) != manifest.get("completed_fold_count"):
        raise ArtifactError("Completed fold count mismatch.")
    if (
        summary["completed_fold_count"] != len(folds)
        or summary["planned_fold_count"] != manifest["planned_fold_count"]
    ):
        raise ArtifactError("Manifest/summary count mismatch.")
    return manifest
