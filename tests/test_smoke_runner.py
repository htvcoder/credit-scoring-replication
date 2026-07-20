from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from creditrep.config.exceptions import ConfigError
from creditrep.config.loader import config_hash, load_experiment_config, parse_experiment_config
from creditrep.datasets import load_dataset
from creditrep.evaluation.exceptions import EvaluationError
from creditrep.evaluation.metrics import compute_binary_metrics
from creditrep.evaluation.predictions import (
    prediction_hash,
    validate_positive_class_probabilities,
    validate_prediction_frame,
)
from creditrep.experiments.runner import run_smoke_experiment
from creditrep.models import create_model
from creditrep.preprocessing import build_smoke_preprocessor
from creditrep.splitting import create_split

ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def smoke_config(model_type: str = "logistic_regression", *, output_root: str = "artifacts/experiments") -> dict:
    parameters = (
        {"max_iter": 1000, "solver": "liblinear", "random_state": 42}
        if model_type == "logistic_regression"
        else {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "n_jobs": 1,
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
    )
    return {
        "experiment": {"name": f"smoke_toy_{model_type}", "purpose": "smoke_validation", "publishable": False},
        "dataset": {"id": "TOY"},
        "split": {"strategy": "stratified_holdout", "test_size": 0.25, "random_seed": 42, "shuffle": True},
        "preprocessing": {"mode": "smoke_baseline"},
        "model": {"type": model_type, "parameters": parameters},
        "evaluation": {"classification_threshold": 0.5},
        "output": {"root_dir": output_root},
    }


def prepare_toy_repo(tmp_path: Path) -> None:
    rows = []
    for i in range(24):
        rows.append({"score": float(i), "debt": None if i % 7 == 0 else float(i % 5), "cat": "A", "BAD": 0})
    for i in range(24):
        rows.append({"score": float(100 + i), "debt": None if i % 5 == 0 else float(20 + i % 4), "cat": "B", "BAD": 1})
    data_path = tmp_path / "data" / "raw" / "toy.csv"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(data_path, index=False)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest().upper()
    write_yaml(
        tmp_path / "data" / "datasets.yaml",
        {
            "datasets": {
                "toy": {
                    "id": "toy",
                    "full_name": "Toy",
                    "active_file": "data/raw/toy.csv",
                    "raw_file": "data/raw/toy.csv",
                    "reader": {"type": "csv", "header": True},
                    "target": {"column": "BAD", "mapping_to_binary": {0: 0, 1: 1}},
                    "identifier_columns": [],
                    "ignored_columns": [],
                    "categorical_columns": ["cat"],
                    "numeric_columns": ["score", "debt"],
                    "missing_values": [],
                }
            }
        },
    )
    (tmp_path / "data" / "checksums-sha256.csv").write_text(
        f'"Path","Algorithm","Hash"\n"data/raw/toy.csv","SHA256","{digest}"\n',
        encoding="utf-8",
    )


def test_lr_and_xgboost_configs_load(tmp_path):
    prepare_toy_repo(tmp_path)
    for model_type in ("logistic_regression", "xgboost"):
        config_path = tmp_path / f"{model_type}.yaml"
        write_yaml(config_path, smoke_config(model_type))
        config = load_experiment_config(config_path, repo_root=tmp_path)
        assert config.model_type == model_type
        assert config.publishable is False
        assert config.experiment_purpose == "smoke_validation"


def test_unsupported_model_threshold_publishable_and_invalid_parameter_are_rejected():
    payload = smoke_config()
    payload["model"]["type"] = "random_forest"
    with pytest.raises(ConfigError, match="Unsupported model type"):
        parse_experiment_config(payload)

    payload = smoke_config()
    payload["evaluation"]["classification_threshold"] = 1.2
    with pytest.raises(ConfigError, match="classification_threshold"):
        parse_experiment_config(payload)

    payload = smoke_config()
    payload["experiment"]["publishable"] = True
    with pytest.raises(ConfigError, match="publishable"):
        parse_experiment_config(payload)

    payload = smoke_config()
    payload["model"]["parameters"]["unknown"] = 1
    with pytest.raises(ConfigError, match="unsupported model parameters"):
        parse_experiment_config(payload)


def test_model_seed_conflict_is_rejected():
    payload = smoke_config()
    payload["model"]["parameters"]["random_state"] = 7
    with pytest.raises(ConfigError, match="conflicts"):
        parse_experiment_config(payload)


def test_config_hash_changes_when_model_parameters_change():
    left = parse_experiment_config(smoke_config())
    payload = smoke_config()
    payload["model"]["parameters"]["C"] = 0.5
    right = parse_experiment_config(payload)
    assert config_hash(left) != config_hash(right)


def test_preprocessor_handles_missing_unknown_category_and_train_only_state(tmp_path):
    prepare_toy_repo(tmp_path)
    dataset = load_dataset("TOY", repo_root=tmp_path)
    train = dataset.features.iloc[:30].copy()
    test = dataset.features.iloc[30:].copy()
    test.loc[test.index[0], "cat"] = "TEST_ONLY"
    test.loc[test.index[0], "score"] = 99999
    preprocessor = build_smoke_preprocessor(
        features_columns=list(dataset.features.columns),
        dataset_metadata=dataset.metadata,
        model_type="logistic_regression",
    )

    train_matrix = preprocessor.fit_transform(train)
    test_matrix = preprocessor.transform(test)
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]

    assert numeric_imputer.statistics_[0] != 99999
    assert "TEST_ONLY" not in set(onehot.categories_[0])
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(test_matrix).all()


def test_models_fit_and_probability_contract(tmp_path):
    prepare_toy_repo(tmp_path)
    dataset = load_dataset("TOY", repo_root=tmp_path)
    split = create_split(dataset, test_size=0.25, random_seed=42)
    for model_type in ("logistic_regression", "xgboost"):
        config = parse_experiment_config(smoke_config(model_type))
        preprocessor = build_smoke_preprocessor(
            features_columns=list(dataset.features.columns),
            dataset_metadata=dataset.metadata,
            model_type=model_type,
        )
        model = create_model(model_type, config.model_parameters)
        train_matrix = preprocessor.fit_transform(split.train_features)
        test_matrix = preprocessor.transform(split.test_features)
        model.fit(train_matrix, split.train_target)
        y_score = validate_positive_class_probabilities(
            model,
            model.predict_proba(test_matrix),
            expected_rows=len(split.test_target),
        )
        assert len(y_score) == len(split.test_target)
        assert ((y_score >= 0) & (y_score <= 1)).all()


def test_probability_validation_rejects_bad_values():
    class BadModel:
        classes_ = np.array([0, 1])

    with pytest.raises(EvaluationError, match="NaN"):
        validate_positive_class_probabilities(BadModel(), np.array([[0.0, np.nan]]), expected_rows=1)
    with pytest.raises(EvaluationError, match="Prediction length"):
        validate_positive_class_probabilities(BadModel(), np.array([[0.2, 0.8]]), expected_rows=2)


def test_metrics_known_values_and_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    y_pred = (y_score >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_true=y_true, y_score=y_score, y_pred=y_pred, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(np.mean((y_score - y_true) ** 2))
    assert metrics["classification_threshold"] == 0.5
    assert all(math.isfinite(value) for value in metrics.values() if isinstance(value, float))


def test_prediction_frame_validation_and_hash():
    frame = pd.DataFrame(
        {
            "row_position": [1, 3],
            "partition": ["test", "test"],
            "y_true": [0, 1],
            "y_score": [0.2, 0.8],
            "y_pred": [0, 1],
        }
    )
    validate_prediction_frame(frame, expected_rows=2)
    assert prediction_hash(frame, split_hash="s", model_config_hash="m") == prediction_hash(
        frame.iloc[::-1],
        split_hash="s",
        model_config_hash="m",
    )
    bad = frame.copy()
    bad.loc[0, "y_score"] = 2.0
    with pytest.raises(EvaluationError, match="y_score"):
        validate_prediction_frame(bad)


def test_lr_runner_is_deterministic_and_writes_artifacts(tmp_path):
    prepare_toy_repo(tmp_path)
    config_path = tmp_path / "configs" / "smoke_lr.yaml"
    write_yaml(config_path, smoke_config("logistic_regression"))

    first = run_smoke_experiment(config_path, repo_root=tmp_path)
    second = run_smoke_experiment(config_path, repo_root=tmp_path)

    assert first.split_hash == second.split_hash
    assert first.prediction_hash == second.prediction_hash
    assert first.metrics == second.metrics
    manifest = json.loads((first.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((first.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    model_metadata = json.loads((first.artifact_dir / "model_metadata.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(first.artifact_dir / "predictions.csv")

    assert manifest["publishable"] is False
    assert manifest["result_scope"] == "smoke_validation"
    assert manifest["model"]["type"] == "logistic_regression"
    assert metrics["publishable"] is False
    assert model_metadata["library_versions"]["scikit_learn"]
    assert list(predictions.columns) == ["row_position", "partition", "y_true", "y_score", "y_pred"]
    assert len(predictions) == manifest["split"]["test_row_count"]
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in first.artifact_dir.glob("*.json"))
    assert "D:\\" not in artifact_text
    assert "score" not in predictions.columns


def test_xgboost_runner_completes_on_fixture(tmp_path):
    prepare_toy_repo(tmp_path)
    config_path = tmp_path / "configs" / "smoke_xgb.yaml"
    write_yaml(config_path, smoke_config("xgboost"))

    result = run_smoke_experiment(config_path, repo_root=tmp_path)

    assert result.model_type == "xgboost"
    assert result.manifest["status"] == "completed"
    assert result.metrics["test_row_count"] == 12


def test_cli_success_invalid_config_and_no_raw_records(tmp_path):
    prepare_toy_repo(tmp_path)
    config_path = tmp_path / "configs" / "smoke_lr.yaml"
    write_yaml(config_path, smoke_config("logistic_regression"))
    script = ROOT / "scripts" / "run_experiment.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["publishable"] is False
    assert summary["model"] == "logistic_regression"
    assert "score" not in completed.stdout

    bad_path = tmp_path / "bad.yaml"
    payload = smoke_config("logistic_regression")
    payload["model"]["type"] = "bad"
    write_yaml(bad_path, payload)
    failed = subprocess.run(
        [sys.executable, str(script), "--config", str(bad_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
