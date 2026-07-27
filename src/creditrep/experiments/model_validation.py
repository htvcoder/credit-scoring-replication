"""P5C nested-CV runner with per-fold resume/retry orchestration."""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.model_validation import (ArtifactValidationError, initialise_experiment, reconcile_summary,
    resolve_failure_artifact, validate_failure_artifact, validate_fold, write_failure_artifact,
    write_fold_artifact)
from creditrep.config.loader import canonical_json, sha256_canonical
from creditrep.config.model_validation import ModelValidationConfig
from creditrep.datasets.models import LoadedDataset
from creditrep.evaluation.predictions import build_prediction_frame
from creditrep.experiments.nested_cv import _fit_preprocessing
from creditrep.metrics.registry import compute_configured_metric, get_metric_specification
from creditrep.models import build_model_metadata, create_model, positive_class_probabilities
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import NestedCVDefinition


@dataclass(frozen=True)
class ModelValidationResult:
    folds: dict[str, dict[str, Any]]
    summary: dict[str, Any]


class FoldStageError(RuntimeError):
    """Preserve the execution boundary that produced a fold failure."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def _at_stage(stage: str, operation):
    try:
        return operation()
    except FoldStageError:
        raise
    except Exception as exc:
        raise FoldStageError(stage, exc) from exc


def _score(config: ModelValidationConfig, y_true, y_score) -> float:
    metric = next(item for item in config.metrics if item.metric_id == config.optimization_metric)
    result = compute_configured_metric(metric, y_true, y_score)
    if result.status != "valid" or result.value is None:
        return float("-inf")
    return float(result.value) if get_metric_specification(metric.metric_id, parameters=metric.parameters).direction == "maximize" else -float(result.value)


def _run_fold(*, config: ModelValidationConfig, dataset: LoadedDataset, outer: Any, model_id: str, candidates: tuple[dict[str, Any], ...], protocol_config: ProtocolAConfig) -> dict[str, Any]:
    candidate_rows = []
    candidate_warnings = []
    for index, parameters in enumerate(candidates):
        scores = []
        try:
            for inner in outer.inner_folds:
                pipeline, train, validation = _at_stage("preprocessing", lambda: _fit_preprocessing(dataset, train_indices=inner.train_indices, transform_indices=inner.validation_indices, protocol_config=protocol_config))
                def evaluate_inner():
                    estimator = create_model(model_id, parameters, random_seed=inner.seed)
                    estimator.fit(train, dataset.target.iloc[list(inner.train_indices)])
                    score = positive_class_probabilities(estimator, estimator.predict_proba(validation), expected_rows=len(validation))
                    return _score(config, dataset.target.iloc[list(inner.validation_indices)], score)
                scores.append(_at_stage("inner_tuning", evaluate_inner))
            mean_score = sum(scores) / len(scores)
            if not pd.notna(mean_score) or mean_score in (float("inf"), float("-inf")):
                raise FoldStageError("inner_tuning", ValueError("candidate produced no valid finite inner score"))
            candidate_rows.append({"index": index, "parameters": dict(parameters), "candidate_hash": sha256_canonical(parameters), "inner_scores": scores, "mean_score": mean_score})
        except FoldStageError as exc:
            if exc.stage == "preprocessing":
                raise
            candidate_warnings.append({"candidate_hash": sha256_canonical(parameters), "exception_type": type(exc.cause).__name__, "message": str(exc.cause).splitlines()[0][:200]})
    if not candidate_rows:
        raise FoldStageError("inner_tuning", ValueError("all model candidates failed or produced invalid scores"))
    selected = sorted(candidate_rows, key=lambda item: (-item["mean_score"], canonical_json(item["parameters"]), item["index"]))[0]
    prep_start = time.perf_counter()
    pipeline, train, test = _at_stage("preprocessing", lambda: _fit_preprocessing(dataset, train_indices=outer.train_indices, transform_indices=outer.test_indices, protocol_config=protocol_config))
    preprocessing_seconds = time.perf_counter() - prep_start
    fit_start = time.perf_counter()
    def refit_outer():
        estimator = create_model(model_id, selected["parameters"], random_seed=outer.seed)
        estimator.fit(train, dataset.target.iloc[list(outer.train_indices)])
        return estimator
    estimator = _at_stage("outer_refit", refit_outer)
    fit_seconds = time.perf_counter() - fit_start
    prediction_start = time.perf_counter()
    y_score = _at_stage("prediction", lambda: positive_class_probabilities(estimator, estimator.predict_proba(test), expected_rows=len(test)))
    prediction_seconds = time.perf_counter() - prediction_start
    prediction = _at_stage("prediction", lambda: build_prediction_frame(row_positions=outer.test_indices, y_true=dataset.target.iloc[list(outer.test_indices)], y_score=y_score, threshold=config.threshold))
    metrics = _at_stage("metrics", lambda: [compute_configured_metric(metric, prediction["y_true"], prediction["y_score"]).to_dict() for metric in config.metrics])
    return {"outer_fold_id": outer.outer_fold_id, "model_id": model_id, "fold_hash": outer.split_hash, "publishable": False, "result_scope": "model_validation", "selected_candidate": selected, "preprocessing": pipeline.get_metadata(), "model_metadata": build_model_metadata(estimator, model_id=model_id, configured_hyperparameters=selected["parameters"], random_seed=outer.seed, fit_duration_seconds=fit_seconds, prediction_duration_seconds=prediction_seconds), "metrics": metrics, "predictions": prediction, "train_count": len(outer.train_indices), "test_count": len(outer.test_indices), "train_class_counts": outer.train_class_counts, "test_class_counts": outer.test_class_counts, "inner_fold_count": len(outer.inner_folds), "warnings": candidate_warnings, "timings": {"preprocessing_seconds": preprocessing_seconds, "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds}}


def run_model_validation(*, config: ModelValidationConfig, dataset: LoadedDataset, nested_cv: NestedCVDefinition, protocol_config: ProtocolAConfig) -> ModelValidationResult:
    """In-memory compatibility runner; callers needing resume use ``run_folded_model_validation``."""
    folds = {}
    for outer in nested_cv.outer_folds:
        for model_id, candidates in sorted(config.model_candidates.items()):
            key = f"{outer.outer_fold_id}__{model_id}"
            folds[key] = _run_fold(config=config, dataset=dataset, outer=outer, model_id=model_id, candidates=candidates, protocol_config=protocol_config)
    return ModelValidationResult(folds=folds, summary={"planned_fold_count": len(nested_cv.outer_folds) * len(config.model_candidates), "completed_fold_count": len(folds), "failed_fold_count": 0, "publishable": False, "result_scope": "model_validation", "config_hash": config.config_hash})


def run_folded_model_validation(*, config: ModelValidationConfig, dataset: LoadedDataset, nested_cv: NestedCVDefinition, protocol_config: ProtocolAConfig, output_root: Path | str, dataset_checksum: str, repo_root: Path | str | None = None, resume: bool = False, fail_fast: bool = False) -> tuple[Path, dict[str, Any]]:
    """Execute model×outer-fold units independently, preserving valid completed work."""
    experiment_id = f"{config.experiment_name}-{config.config_hash[:12]}"
    units = {f"{outer.outer_fold_id}__{model_id}": {"fold_hash": outer.split_hash, "model_id": model_id} for outer in nested_cv.outer_folds for model_id in config.model_candidates}
    root = initialise_experiment(root=output_root, config=config, experiment_id=experiment_id, dataset_checksum=dataset_checksum, planned_fold_count=len(units), repo_root=repo_root)
    # Interrupted temporary writes are never valid fold artifacts and are rebuilt fresh.
    for temporary in sorted((root / "folds").glob(".tmp-*")):
        if temporary.is_dir():
            shutil.rmtree(temporary)
    resumed = retried = 0
    for outer in nested_cv.outer_folds:
        for model_id, candidates in sorted(config.model_candidates.items()):
            fold_id = f"{outer.outer_fold_id}__{model_id}"
            fold_path = root / "folds" / fold_id
            failure_path = root / "failures" / f"{fold_id}.json"
            if fold_path.exists():
                try:
                    validate_fold(fold_path, config_hash=config.config_hash, dataset_checksum=dataset_checksum, fold_hash=outer.split_hash, model_id=model_id, experiment_id=experiment_id, dataset_id=config.dataset_id)
                    if not resume:
                        raise ArtifactError(f"Completed fold exists: {fold_id}; use --resume.")
                    resumed += 1
                    continue
                except ArtifactError as exc:
                    if "Completed fold exists" in str(exc):
                        raise
                    if "mismatch" in str(exc).lower() or "schema version" in str(exc).lower():
                        raise ArtifactError(f"Fold provenance mismatch for {fold_id}; use a new experiment ID.") from exc
                    # Preserve corrupt evidence; never overwrite it in place.
                    quarantine = root / "corrupt" / f"{fold_id}-{int(time.time() * 1000)}"
                    quarantine.parent.mkdir(exist_ok=True)
                    fold_path.replace(quarantine)
            if failure_path.exists():
                failure = validate_failure_artifact(failure_path, config_hash=config.config_hash)
                if failure.get("fold_hash") != outer.split_hash or failure.get("model_id") != model_id:
                    raise ArtifactError(f"Failure artifact provenance mismatch for {fold_id}.")
                if not resume:
                    raise ArtifactError(f"Failed fold exists: {fold_id}; use --resume to retry.")
                retried += 1
            try:
                fold = _run_fold(config=config, dataset=dataset, outer=outer, model_id=model_id, candidates=candidates, protocol_config=protocol_config)
                write_fold_artifact(experiment_root=root, config=config, dataset_checksum=dataset_checksum, fold_id=fold_id, fold=fold)
                resolve_failure_artifact(root, fold_id)
            except Exception as exc:
                temporary = root / "folds" / f".tmp-{fold_id}"
                if temporary.exists():
                    shutil.rmtree(temporary)
                stage = exc.stage if isinstance(exc, FoldStageError) else "artifact_validation" if isinstance(exc, ArtifactValidationError) else "artifact_write"
                cause = exc.cause if isinstance(exc, FoldStageError) else exc
                write_failure_artifact(root=root, experiment_id=experiment_id, dataset_id=config.dataset_id, model_id=model_id, fold_id=fold_id, fold_hash=outer.split_hash, stage=stage, exception=cause, config_hash=config.config_hash)
                summary = reconcile_summary(experiment_root=root, planned_units=units, config=config, dataset_checksum=dataset_checksum)
                if fail_fast:
                    raise RuntimeError(f"Fold failed: {fold_id}") from exc
    summary = reconcile_summary(experiment_root=root, planned_units=units, config=config, dataset_checksum=dataset_checksum)
    summary["resumed_skipped_fold_count"] = resumed
    summary["retried_fold_count"] = retried
    # Write the counters atomically after reconciliation.
    import json
    temp = root / "summary.tmp"
    temp.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(root / "summary.json")
    return root, summary
