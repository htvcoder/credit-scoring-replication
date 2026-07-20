"""P2C smoke experiment orchestration."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from creditrep.artifacts.writer import create_smoke_experiment_artifact
from creditrep.checksums import get_dataset_checksum
from creditrep.config.loader import config_hash, load_experiment_config
from creditrep.datasets import load_dataset
from creditrep.datasets.registry import find_repo_root
from creditrep.evaluation.metrics import compute_binary_metrics
from creditrep.evaluation.predictions import (
    build_prediction_frame,
    prediction_hash,
    validate_positive_class_probabilities,
)
from creditrep.experiments.exceptions import ExperimentError
from creditrep.experiments.models import ExperimentRunResult
from creditrep.models import create_model
from creditrep.preprocessing import build_smoke_preprocessor
from creditrep.splitting import create_split


def _require_smoke_config(config) -> None:
    if config.model_type is None:
        raise ExperimentError("Smoke experiment config must include model.type.")
    if config.experiment_purpose != "smoke_validation":
        raise ExperimentError("Smoke experiment purpose must be smoke_validation.")
    if config.publishable is not False:
        raise ExperimentError("Smoke experiment must set publishable: false.")
    if config.classification_threshold is None:
        raise ExperimentError("Smoke experiment config must include evaluation.classification_threshold.")


def _transformed_feature_count(pipeline: Pipeline, features) -> int | None:
    try:
        transformed = pipeline.named_steps["preprocessor"].transform(features.iloc[:1])
    except Exception:
        return None
    return int(transformed.shape[1])


def _validate_probability_estimator(pipeline: Pipeline) -> None:
    if not hasattr(pipeline, "predict_proba"):
        raise ExperimentError("Model pipeline must expose predict_proba.")


def run_smoke_experiment(
    config_path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> ExperimentRunResult:
    """Run one non-publishable P2C smoke experiment."""

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    config = load_experiment_config(config_path, repo_root=root)
    _require_smoke_config(config)

    dataset = load_dataset(config.dataset_id, repo_root=root)
    checksum = get_dataset_checksum(dataset.dataset_id, dataset.metadata["source_file"], repo_root=root)
    dataset.metadata["checksum_sha256"] = checksum.actual_sha256
    split = create_split(
        dataset,
        strategy=config.split_strategy,
        test_size=config.test_size,
        random_seed=config.random_seed,
        shuffle=config.shuffle,
    )

    preprocessor = build_smoke_preprocessor(
        features_columns=list(dataset.features.columns),
        dataset_metadata=dataset.metadata,
        model_type=config.model_type,
    )
    model = create_model(config.model_type, config.model_parameters)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    _validate_probability_estimator(pipeline)

    fit_warnings: list[str] = []
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(split.train_features, split.train_target)
    fit_duration = time.perf_counter() - fit_start
    fit_warnings = [f"{warning.category.__name__}: {warning.message}" for warning in caught]

    predict_start = time.perf_counter()
    probabilities = pipeline.predict_proba(split.test_features)
    y_score = validate_positive_class_probabilities(
        pipeline.named_steps["model"],
        np.asarray(probabilities),
        expected_rows=len(split.test_target),
    )
    predict_duration = time.perf_counter() - predict_start
    predictions = build_prediction_frame(
        row_positions=split.test_indices,
        y_true=split.test_target,
        y_score=y_score,
        threshold=config.classification_threshold,
    )
    pred_hash = prediction_hash(
        predictions,
        split_hash=split.split_hash,
        model_config_hash=config_hash(config),
    )
    metrics = compute_binary_metrics(
        y_true=predictions["y_true"],
        y_score=predictions["y_score"].to_numpy(),
        y_pred=predictions["y_pred"].to_numpy(),
        threshold=config.classification_threshold,
    )
    model_metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "model_type": config.model_type,
        "model_parameters": config.model_parameters,
        "preprocessing_mode": config.preprocessing_mode,
        "numeric_preprocessing_steps": ["median_imputer", "standard_scaler"]
        if config.model_type == "logistic_regression"
        else ["median_imputer"],
        "categorical_preprocessing_steps": ["most_frequent_imputer", "one_hot_encoder_handle_unknown_ignore"],
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__ if config.model_type == "xgboost" else None,
        },
        "fit_duration_seconds": float(fit_duration),
        "prediction_duration_seconds": float(predict_duration),
        "random_seed": config.random_seed,
        "feature_count_before_preprocessing": int(split.train_features.shape[1]),
        "transformed_feature_count": _transformed_feature_count(pipeline, split.train_features),
        "fit_warnings": fit_warnings,
        "model_artifact_saved": False,
    }
    artifact_dir, manifest = create_smoke_experiment_artifact(
        config=config,
        dataset=dataset,
        split=split,
        checksum=checksum,
        metrics=metrics,
        predictions=predictions,
        model_metadata=model_metadata,
        prediction_hash=pred_hash,
        repo_root=root,
    )
    return ExperimentRunResult(
        experiment_id=manifest["experiment_id"],
        artifact_dir=artifact_dir,
        dataset_id=dataset.dataset_id,
        model_type=config.model_type,
        split_hash=split.split_hash,
        prediction_hash=pred_hash,
        metrics=metrics,
        manifest=manifest,
    )
