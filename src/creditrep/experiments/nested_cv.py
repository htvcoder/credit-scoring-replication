"""P3C nested CV orchestration and tuning-isolation harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from creditrep.config.loader import sha256_canonical
from creditrep.config.nested import NestedCVConfig
from creditrep.datasets.models import LoadedDataset
from creditrep.preprocessing import ProtocolAConfig, ProtocolAPreprocessingPipeline
from creditrep.splitting.nested import NestedCVDefinition, OuterFoldDefinition


@dataclass(frozen=True)
class NestedCVValidationResult:
    nested_cv: NestedCVDefinition
    inner_preprocessing: dict[str, dict[str, Any]]
    outer_preprocessing: dict[str, dict[str, Any]]
    tuning_summaries: dict[str, dict[str, Any]]
    pipeline_instance_ids: dict[str, int]


class FakeCandidateEstimator:
    """Tiny deterministic estimator used only to prove fresh fit isolation."""

    fit_counter = 0

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters = dict(parameters)
        self.fit_id: int | None = None
        self.fitted_rows = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FakeCandidateEstimator":
        FakeCandidateEstimator.fit_counter += 1
        self.fit_id = FakeCandidateEstimator.fit_counter
        self.fitted_rows = len(X)
        return self

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        # Deterministic infrastructure-only score. It intentionally depends on
        # inner validation data and candidate parameters, never outer test data.
        bias = float(self.parameters.get("bias", 0.0))
        return float(y.mean()) - abs(bias)


def _candidate_hash(candidate: dict[str, Any]) -> str:
    return sha256_canonical(candidate)


def _fit_preprocessing(
    dataset: LoadedDataset,
    *,
    train_indices: tuple[int, ...],
    transform_indices: tuple[int, ...],
    protocol_config: ProtocolAConfig,
) -> tuple[ProtocolAPreprocessingPipeline, pd.DataFrame, pd.DataFrame]:
    pipeline = ProtocolAPreprocessingPipeline(
        dataset_metadata=dataset.metadata,
        config=protocol_config,
    )
    X_train = dataset.features.iloc[list(train_indices)].copy(deep=True)
    y_train = dataset.target.iloc[list(train_indices)].copy(deep=True)
    X_transform = dataset.features.iloc[list(transform_indices)].copy(deep=True)
    pipeline.fit(X_train, y_train)
    train_matrix = pipeline.transform(X_train)
    transform_matrix = pipeline.transform(X_transform)
    return pipeline, train_matrix, transform_matrix


def _evaluate_candidates(
    config: NestedCVConfig,
    dataset: LoadedDataset,
    outer: OuterFoldDefinition,
    inner_matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, Any]:
    candidate_results: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(config.candidates):
        fold_scores: list[float] = []
        fit_ids: list[int] = []
        for inner in outer.inner_folds:
            train_matrix, validation_matrix = inner_matrices[inner.inner_fold_id]
            estimator = FakeCandidateEstimator(candidate)
            estimator.fit(train_matrix, dataset.target.iloc[list(inner.train_indices)])
            fit_ids.append(int(estimator.fit_id or -1))
            score = estimator.score(validation_matrix, dataset.target.iloc[list(inner.validation_indices)])
            fold_scores.append(float(score))
        candidate_results.append(
            {
                "candidate_index": candidate_index,
                "parameters": dict(candidate),
                "inner_scores": fold_scores,
                "mean_inner_score": float(sum(fold_scores) / len(fold_scores)),
                "estimator_fit_ids": fit_ids,
            }
        )

    ranked = sorted(
        candidate_results,
        key=lambda item: (-item["mean_inner_score"], item["candidate_index"]),
    )
    best = ranked[0]
    return {
        "outer_fold_id": outer.outer_fold_id,
        "selection_rule": "highest_mean_inner_score_then_config_order",
        "candidate_hashes": [
            {"candidate_index": item["candidate_index"], "candidate_hash": _candidate_hash(item["parameters"])}
            for item in candidate_results
        ],
        "result_scope": "preprocessing_validation",
        "publishable": False,
        "candidate_results": candidate_results,
        "selected_candidate_index": best["candidate_index"],
        "selected_parameters": dict(best["parameters"]),
        "outer_test_metric": None,
    }


def run_nested_cv_validation(
    *,
    config: NestedCVConfig,
    dataset: LoadedDataset,
    nested_cv: NestedCVDefinition,
    protocol_config: ProtocolAConfig,
) -> NestedCVValidationResult:
    """Fit per-fold preprocessing and infrastructure-only tuning summaries."""

    inner_preprocessing: dict[str, dict[str, Any]] = {}
    outer_preprocessing: dict[str, dict[str, Any]] = {}
    tuning_summaries: dict[str, dict[str, Any]] = {}
    pipeline_instance_ids: dict[str, int] = {}

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

        final_pipeline, _, _ = _fit_preprocessing(
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

    return NestedCVValidationResult(
        nested_cv=nested_cv,
        inner_preprocessing=inner_preprocessing,
        outer_preprocessing=outer_preprocessing,
        tuning_summaries=tuning_summaries,
        pipeline_instance_ids=pipeline_instance_ids,
    )
