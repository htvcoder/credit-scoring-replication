"""Artifact-contract acceptance tests for the P5C fixture runner."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.model_validation import validate_failure_artifact, validate_fold, validate_model_validation_artifact
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.experiments.model_validation import run_folded_model_validation
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition


def _config(models=None):
    return parse_model_validation_config({"experiment": {"name": "artifact_fixture", "publishable": False, "result_scope": "model_validation"}, "dataset": {"id": "TOY"}, "output": {"root_dir": "ignored"}, "models": models or {"logistic_regression": [{"max_iter": 30, "solver": "liblinear"}]}, "metrics": [{"id": "roc_auc"}, {"id": "emp"}], "optimization_metric": "roc_auc", "random_seed": 17})


def _dataset():
    return LoadedDataset("TOY", pd.DataFrame({"x": range(24)}), pd.Series([0, 1] * 12), {"numeric_columns": ["x"], "categorical_columns": [], "source_file": "fixture.csv", "row_count": 24, "feature_count": 1}, Path("fixture.csv"))


def _artifact(tmp_path, models=None):
    dataset, config = _dataset(), _config(models)
    nested = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    root, _ = run_folded_model_validation(config=config, dataset=dataset, nested_cv=nested, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    return root, config


@pytest.mark.parametrize("name", ["metrics.json", "predictions.csv", "complete.json"])
def test_fold_validator_rejects_missing_required_artifact(tmp_path, name):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    (fold / name).unlink()
    with pytest.raises(ArtifactError, match="Incomplete"):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


@pytest.mark.parametrize("mutation", ["raw_column", "duplicate", "invalid_probability", "row_count"])
def test_fold_validator_rejects_prediction_contract_corruption(tmp_path, mutation):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    predictions = pd.read_csv(fold / "predictions.csv")
    if mutation == "raw_column":
        predictions["feature_1"] = 7
    elif mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif mutation == "invalid_probability":
        predictions.loc[0, "y_score"] = 1.5
    else:
        metadata = json.loads((fold / "fold_metadata.json").read_text(encoding="utf-8"))
        metadata["test_count"] += 1
        (fold / "fold_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    predictions.to_csv(fold / "predictions.csv", index=False)
    with pytest.raises(ArtifactError, match="Invalid predictions|Invalid prediction probabilities|row-count"):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


def test_decision_tree_file_requires_cart_provenance(tmp_path):
    root, config = _artifact(tmp_path, {"decision_tree": [{"max_depth": 2}]})
    fold = next((root / "folds").iterdir())
    metadata = json.loads((fold / "model_metadata.json").read_text(encoding="utf-8"))
    metadata["algorithm"] = "c45"
    (fold / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError, match="CART provenance"):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


@pytest.mark.parametrize("name", ["complete.json", "fold_metadata.json", "tuning.json", "preprocessing.json", "model_metadata.json", "metrics.json"])
@pytest.mark.parametrize("content", ["{", "", "[]"])
def test_artifact_validator_rejects_malformed_json_files(tmp_path, name, content):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    (fold / name).write_text(content, encoding="utf-8")
    with pytest.raises(ArtifactError, match="Malformed|mismatch|invalid"):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


@pytest.mark.parametrize("content", ["", "not,a,contract\n1,2", "row_position,outer_repeat\n1", "row_position,outer_repeat,outer_fold,partition,y_true,y_score,y_pred\n0,0,0,test,0,bad,0"])
def test_prediction_validator_rejects_malformed_csv_contract(tmp_path, content):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    (fold / "predictions.csv").write_text(content, encoding="utf-8")
    with pytest.raises(ArtifactError, match="Malformed|Invalid"):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


@pytest.mark.parametrize("column,value", [("y_true", -1), ("y_true", 2), ("y_true", 0.5), ("y_true", "bad"), ("y_pred", -1), ("y_pred", 2), ("y_pred", 0.5), ("y_pred", "good")])
def test_prediction_validator_rejects_invalid_binary_values(tmp_path, column, value):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    predictions = pd.read_csv(fold / "predictions.csv")
    predictions[column] = predictions[column].astype("object")
    predictions.loc[0, column] = value
    predictions.to_csv(fold / "predictions.csv", index=False)
    with pytest.raises(ArtifactError, match=column):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


@pytest.mark.parametrize("field,value", [("experiment_id", "other"), ("dataset_id", "OTHER"), ("dataset_checksum", "other"), ("config_hash", "other"), ("fold_id", "other"), ("fold_hash", "other"), ("model_id", "decision_tree"), ("schema_version", "0"), ("publishable", True), ("result_scope", "scientific")])
def test_resume_validator_rejects_all_critical_fold_provenance_mismatches(tmp_path, field, value):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    path = fold / "fold_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture", experiment_id=root.name, dataset_id="TOY")


def test_prediction_fold_provenance_and_partition_must_match_metadata(tmp_path):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    predictions = pd.read_csv(fold / "predictions.csv")
    predictions.loc[0, "outer_fold"] = 99
    predictions.loc[1, "partition"] = "train"
    predictions.to_csv(fold / "predictions.csv", index=False)
    with pytest.raises(ArtifactError, match="predictions|provenance"):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_prediction_validator_rejects_nonfinite_or_out_of_range_probabilities(tmp_path, value):
    root, config = _artifact(tmp_path)
    fold = next((root / "folds").iterdir())
    predictions = pd.read_csv(fold / "predictions.csv")
    predictions.loc[0, "y_score"] = value
    predictions.to_csv(fold / "predictions.csv", index=False)
    with pytest.raises(ArtifactError, match="predictions|probabilities"):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")


def test_full_artifact_validator_rejects_manifest_config_and_fold_mismatch(tmp_path):
    root, config = _artifact(tmp_path)
    config_path = root / "config.yaml"
    original = config_path.read_text(encoding="utf-8")
    config_path.write_text(original + "\nrandom_seed: 999\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="Manifest/config"):
        validate_model_validation_artifact(root, config_hash=config.config_hash, dataset_checksum="fixture")
    config_path.write_text(original, encoding="utf-8")
    fold = next((root / "folds").iterdir())
    metadata_path = fold / "fold_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["outer_repeat"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError, match="provenance"):
        validate_model_validation_artifact(root, config_hash=config.config_hash, dataset_checksum="fixture")


@pytest.mark.parametrize("content", ["{", "", "[]"])
def test_full_artifact_validator_rejects_malformed_manifest_and_summary(tmp_path, content):
    root, config = _artifact(tmp_path)
    for name in ("manifest.json", "summary.json"):
        original = (root / name).read_text(encoding="utf-8")
        (root / name).write_text(content, encoding="utf-8")
        with pytest.raises(ArtifactError, match="Malformed"):
            validate_model_validation_artifact(root, config_hash=config.config_hash, dataset_checksum="fixture")
        (root / name).write_text(original, encoding="utf-8")


@pytest.mark.parametrize("mutation", ["missing_stage", "invalid_stage", "invalid_attempt", "wrong_root"])
def test_corrupt_failure_artifact_is_never_silently_trusted(tmp_path, mutation):
    root, config = _artifact(tmp_path)
    failure = root / "failures" / "bad.json"
    failure.parent.mkdir(exist_ok=True)
    payload = {"schema_version": "1.1", "experiment_id": root.name, "dataset_id": "TOY", "model_id": "logistic_regression", "fold_id": "bad", "fold_hash": "hash", "stage": "metrics", "exception_type": "ValueError", "message": "bad", "retryable": True, "config_hash": config.config_hash, "timestamp_utc": "2026-01-01T00:00:00Z", "first_failure_timestamp_utc": "2026-01-01T00:00:00Z", "cleanup_status": "completed", "attempt": 1, "resolved": False, "publishable": False, "result_scope": "model_validation"}
    if mutation == "missing_stage":
        payload.pop("stage")
    elif mutation == "invalid_stage":
        payload["stage"] = "unknown"
    elif mutation == "invalid_attempt":
        payload["attempt"] = 0
    if mutation == "wrong_root":
        failure.write_text("[]", encoding="utf-8")
    else:
        failure.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError):
        validate_failure_artifact(failure, config_hash=config.config_hash)
