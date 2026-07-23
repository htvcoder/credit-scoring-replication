"""Phase 4 metric-validation harness built on the P3C nested CV foundation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from creditrep.config.loader import sha256_canonical
from creditrep.config.metric_validation import MetricValidationConfig
from creditrep.datasets.models import LoadedDataset
from creditrep.evaluation.predictions import validate_positive_class_probabilities
from creditrep.metrics import MetricResult, compute_roc_auc
from creditrep.metrics.registry import MetricConfig, compute_configured_metric, metric_config_hash
from creditrep.preprocessing import ProtocolAConfig, ProtocolAPreprocessingPipeline
from creditrep.splitting.nested import NestedCVDefinition, OuterFoldDefinition


@dataclass(frozen=True)
class MetricValidationResult:
    nested_cv: NestedCVDefinition
    inner_preprocessing: dict[str, dict[str, Any]]
    outer_preprocessing: dict[str, dict[str, Any]]
    tuning_summaries: dict[str, dict[str, Any]]
    pipeline_instance_ids: dict[str, int]
    fold_metrics: dict[str, tuple[MetricResult, ...]]
    metric_summary: dict[str, dict[str, Any]]
    prediction_summaries: dict[str, dict[str, Any]]
    metric_config_hash: str


class DeterministicProbabilityEstimator:
    """Validation-only estimator used to exercise the metric pipeline without Phase 5 models."""

    fit_counter = 0

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters = dict(parameters)
        self.fit_id: int | None = None
        self.classes_ = np.array([0, 1], dtype=int)
        self._feature_order: list[str] = []
        self._weights: np.ndarray | None = None
        self._intercept = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DeterministicProbabilityEstimator":
        DeterministicProbabilityEstimator.fit_counter += 1
        self.fit_id = DeterministicProbabilityEstimator.fit_counter
        frame = X.astype(float)
        y_array = np.asarray(y, dtype=int)
        if set(np.unique(y_array)) != {0, 1}:
            raise ValueError("DeterministicProbabilityEstimator requires binary classes {0,1}.")
        bad_mean = frame.loc[y_array == 1].mean(axis=0).to_numpy(dtype=float)
        good_mean = frame.loc[y_array == 0].mean(axis=0).to_numpy(dtype=float)
        weights = bad_mean - good_mean
        if np.allclose(weights, 0.0):
            weights = np.ones(frame.shape[1], dtype=float)
        bias = float(self.parameters.get("bias", 0.0))
        midpoint = (bad_mean + good_mean) / 2.0
        self._weights = weights
        self._intercept = float(-(midpoint @ weights) + bias)
        self._feature_order = list(frame.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._weights is None:
            raise ValueError("Estimator must be fit before predict_proba.")
        frame = X.loc[:, self._feature_order].astype(float)
        linear = frame.to_numpy(dtype=float) @ self._weights + self._intercept
        linear = np.clip(linear, -30.0, 30.0)
        positive = 1.0 / (1.0 + np.exp(-linear))
        return np.column_stack([1.0 - positive, positive])


def _fit_preprocessing(
    dataset: LoadedDataset,
    *,
    train_indices: tuple[int, ...],
    transform_indices: tuple[int, ...],
    protocol_config: ProtocolAConfig,
) -> tuple[ProtocolAPreprocessingPipeline, pd.DataFrame, pd.DataFrame]:
    pipeline = ProtocolAPreprocessingPipeline(dataset_metadata=dataset.metadata, config=protocol_config)
    X_train = dataset.features.iloc[list(train_indices)].copy(deep=True)
    y_train = dataset.target.iloc[list(train_indices)].copy(deep=True)
    X_transform = dataset.features.iloc[list(transform_indices)].copy(deep=True)
    pipeline.fit(X_train, y_train)
    train_matrix = pipeline.transform(X_train)
    transform_matrix = pipeline.transform(X_transform)
    return pipeline, train_matrix, transform_matrix


def _score_candidate(
    candidate: dict[str, Any],
    *,
    train_matrix: pd.DataFrame,
    y_train: pd.Series,
    validation_matrix: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[float, int]:
    estimator = DeterministicProbabilityEstimator(candidate)
    estimator.fit(train_matrix, y_train)
    probabilities = estimator.predict_proba(validation_matrix)
    y_score = validate_positive_class_probabilities(estimator, probabilities, expected_rows=len(validation_matrix))
    result = compute_roc_auc(y_validation, y_score)
    value = -math.inf if result.status != "valid" or result.value is None else float(result.value)
    return value, int(estimator.fit_id or -1)


def _evaluate_candidates(
    config: MetricValidationConfig,
    dataset: LoadedDataset,
    outer: OuterFoldDefinition,
    inner_matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(config.candidates):
        fold_scores: list[float | None] = []
        fit_ids: list[int] = []
        for inner in outer.inner_folds:
            train_matrix, validation_matrix = inner_matrices[inner.inner_fold_id]
            score, fit_id = _score_candidate(
                candidate,
                train_matrix=train_matrix,
                y_train=dataset.target.iloc[list(inner.train_indices)],
                validation_matrix=validation_matrix,
                y_validation=dataset.target.iloc[list(inner.validation_indices)],
            )
            fit_ids.append(fit_id)
            fold_scores.append(None if score == -math.inf else float(score))
        valid_scores = [score for score in fold_scores if score is not None]
        mean_inner_score = -math.inf if not valid_scores else float(sum(valid_scores) / len(valid_scores))
        candidate_results.append(
            {
                "candidate_index": candidate_index,
                "parameters": dict(candidate),
                "inner_scores": fold_scores,
                "mean_inner_score": None if mean_inner_score == -math.inf else mean_inner_score,
                "estimator_fit_ids": fit_ids,
            }
        )

    ranked = sorted(
        candidate_results,
        key=lambda item: (
            -math.inf if item["mean_inner_score"] is None else -float(item["mean_inner_score"]),
            item["candidate_index"],
        ),
    )
    best = ranked[0]
    return {
        "outer_fold_id": outer.outer_fold_id,
        "selection_rule": "highest_mean_inner_roc_auc_then_config_order",
        "candidate_hashes": [
            {"candidate_index": item["candidate_index"], "candidate_hash": sha256_canonical(item["parameters"])}
            for item in candidate_results
        ],
        "result_scope": "metric_validation",
        "publishable": False,
        "candidate_results": candidate_results,
        "selected_candidate_index": best["candidate_index"],
        "selected_parameters": dict(best["parameters"]),
        "outer_test_metric": None,
    }


def _aggregate_metrics(metrics: dict[str, tuple[MetricResult, ...]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[MetricResult]] = {}
    for fold_results in metrics.values():
        for result in fold_results:
            grouped.setdefault(result.metric_id, []).append(result)
    summary: dict[str, dict[str, Any]] = {}
    for metric_id, results in sorted(grouped.items()):
        valid_values = [float(result.value) for result in results if result.status == "valid" and result.value is not None]
        warnings = sorted({warning for result in results for warning in result.warnings})
        summary[metric_id] = {
            "metric_id": metric_id,
            "metric_version": results[0].metric_version,
            "direction": results[0].direction,
            "exactness": results[0].exactness,
            "valid_fold_count": sum(result.status == "valid" for result in results),
            "undefined_fold_count": sum(result.status == "undefined" for result in results),
            "unsupported_fold_count": sum(result.status == "unsupported" for result in results),
            "failed_fold_count": sum(result.status == "failed" for result in results),
            "mean": None if not valid_values else float(sum(valid_values) / len(valid_values)),
            "std": None
            if len(valid_values) <= 1
            else float(np.std(np.asarray(valid_values, dtype=float), ddof=0)),
            "warnings": warnings,
        }
    return summary


def run_metric_validation(
    *,
    config: MetricValidationConfig,
    dataset: LoadedDataset,
    nested_cv: NestedCVDefinition,
    protocol_config: ProtocolAConfig,
) -> MetricValidationResult:
    inner_preprocessing: dict[str, dict[str, Any]] = {}
    outer_preprocessing: dict[str, dict[str, Any]] = {}
    tuning_summaries: dict[str, dict[str, Any]] = {}
    pipeline_instance_ids: dict[str, int] = {}
    fold_metrics: dict[str, tuple[MetricResult, ...]] = {}
    prediction_summaries: dict[str, dict[str, Any]] = {}

    for outer in nested_cv.outer_folds:
        inner_matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for inner in outer.inner_folds:
            pipeline, train_matrix, validation_matrix = _fit_preprocessing(
                dataset,
                train_indices=inner.train_indices,
                transform_indices=inner.validation_indices,
                protocol_config=protocol_config,
            )
            before = pipeline.get_metadata()
            pipeline.transform(dataset.features.iloc[list(inner.validation_indices)].copy(deep=True))
            if before != pipeline.get_metadata():
                raise RuntimeError(f"{inner.inner_fold_id}: preprocessing metadata changed after transform.")
            inner_preprocessing[inner.inner_fold_id] = {
                "parent_outer_fold_id": outer.outer_fold_id,
                "inner_fold_id": inner.inner_fold_id,
                "split_hash": inner.split_hash,
                "preprocessing": before,
            }
            pipeline_instance_ids[inner.inner_fold_id] = id(pipeline)
            inner_matrices[inner.inner_fold_id] = (train_matrix, validation_matrix)

        tuning_summaries[outer.outer_fold_id] = _evaluate_candidates(config, dataset, outer, inner_matrices)

        final_pipeline, train_matrix, test_matrix = _fit_preprocessing(
            dataset,
            train_indices=outer.train_indices,
            transform_indices=outer.test_indices,
            protocol_config=protocol_config,
        )
        before = final_pipeline.get_metadata()
        final_pipeline.transform(dataset.features.iloc[list(outer.test_indices)].copy(deep=True))
        if before != final_pipeline.get_metadata():
            raise RuntimeError(f"{outer.outer_fold_id}: final preprocessing metadata changed after transform.")
        outer_preprocessing[outer.outer_fold_id] = {
            "outer_fold_id": outer.outer_fold_id,
            "split_hash": outer.split_hash,
            "selected_parameters": tuning_summaries[outer.outer_fold_id]["selected_parameters"],
            "preprocessing": before,
        }
        pipeline_instance_ids[f"{outer.outer_fold_id}_final"] = id(final_pipeline)

        estimator = DeterministicProbabilityEstimator(tuning_summaries[outer.outer_fold_id]["selected_parameters"])
        estimator.fit(train_matrix, dataset.target.iloc[list(outer.train_indices)])
        probabilities = estimator.predict_proba(test_matrix)
        y_score = validate_positive_class_probabilities(estimator, probabilities, expected_rows=len(test_matrix))
        y_true = dataset.target.iloc[list(outer.test_indices)]
        fold_metrics[outer.outer_fold_id] = tuple(
            compute_configured_metric(metric_config, y_true, y_score) for metric_config in config.metrics
        )
        prediction_summaries[outer.outer_fold_id] = {
            "outer_fold_id": outer.outer_fold_id,
            "fit_id": int(estimator.fit_id or -1),
            "test_row_count": int(len(outer.test_indices)),
            "y_score_min": float(np.min(y_score)),
            "y_score_max": float(np.max(y_score)),
            "y_score_mean": float(np.mean(y_score)),
        }

    return MetricValidationResult(
        nested_cv=nested_cv,
        inner_preprocessing=inner_preprocessing,
        outer_preprocessing=outer_preprocessing,
        tuning_summaries=tuning_summaries,
        pipeline_instance_ids=pipeline_instance_ids,
        fold_metrics=fold_metrics,
        metric_summary=_aggregate_metrics(fold_metrics),
        prediction_summaries=prediction_summaries,
        metric_config_hash=metric_config_hash(config.metrics),
    )
