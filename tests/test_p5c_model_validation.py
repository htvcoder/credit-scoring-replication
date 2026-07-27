from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from creditrep.config.exceptions import ConfigError
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.experiments.model_validation import run_folded_model_validation, run_model_validation
from creditrep.artifacts.model_validation import validate_fold, validate_model_validation_artifact, write_model_validation_artifact
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition


def _config():
    return {"experiment": {"name": "fixture", "publishable": False, "result_scope": "model_validation"}, "dataset": {"id": "TOY"}, "output": {"root_dir": "artifacts/model-validation"}, "models": {"logistic_regression": [{"max_iter": 100, "solver": "liblinear"}], "decision_tree": [{"max_depth": 2}], "random_forest": [{"n_estimators": 4, "n_jobs": 1}], "xgboost": [{"n_estimators": 4, "n_jobs": 1, "tree_method": "hist", "eval_metric": "logloss"}]}, "metrics": [{"id": "roc_auc"}, {"id": "brier_score"}, {"id": "partial_gini"}, {"id": "emp"}], "optimization_metric": "roc_auc", "random_seed": 17}


def _dataset():
    target = pd.Series([0, 1] * 12)
    return LoadedDataset("TOY", pd.DataFrame({"x": list(range(24)), "z": [index % 3 for index in range(24)]}), target, {"numeric_columns": ["x", "z"], "categorical_columns": [], "source_file": "fixture.csv", "row_count": 24, "feature_count": 2}, Path("fixture.csv"))


def test_p5c_config_rejects_emp_and_publishable_output():
    payload = _config()
    payload["optimization_metric"] = "emp"
    with pytest.raises(ConfigError, match="non-EMP"):
        parse_model_validation_config(payload)
    payload = _config()
    payload["experiment"]["publishable"] = True
    with pytest.raises(ConfigError, match="publishable=false"):
        parse_model_validation_config(payload)


def test_four_models_run_outer_test_only_after_inner_selection(tmp_path):
    dataset = _dataset()
    config = parse_model_validation_config(_config())
    definition = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    result = run_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig())
    assert result.summary["completed_fold_count"] == 8
    tree = next(item for item in result.folds.values() if item["model_id"] == "decision_tree")
    assert tree["model_metadata"]["algorithm"] == "cart"
    assert tree["model_metadata"]["deviation_from_paper"] == "c45_to_cart"
    assert set(tree["predictions"].columns) == {"row_position", "partition", "y_true", "y_score", "y_pred"}
    assert tree["publishable"] is False
    assert any(item["metric_id"] == "emp" and item["status"] == "unsupported" for item in tree["metrics"])
    artifact = write_model_validation_artifact(config=config, result=result, output_root=tmp_path)
    manifest = validate_model_validation_artifact(artifact)
    assert manifest["publishable"] is False
    assert (artifact / "folds" / next(key for key, value in result.folds.items() if value["model_id"] == "decision_tree") / "model_metadata.json").exists()


def test_per_fold_resume_skips_valid_completed_units_without_rewriting(tmp_path):
    dataset = _dataset()
    config = parse_model_validation_config(_config())
    definition = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    root, first = run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    assert first["completed_fold_count"] == 8
    prediction = next((root / "folds").glob("*/predictions.csv"))
    before = prediction.read_bytes()
    _, resumed = run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture", resume=True)
    assert resumed["completed_fold_count"] == 8
    assert resumed["resumed_skipped_fold_count"] == 8
    assert prediction.read_bytes() == before


def test_corrupt_fold_is_not_skipped_on_resume(tmp_path):
    dataset = _dataset()
    config = parse_model_validation_config(_config())
    definition = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    root, _ = run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    fold = next((root / "folds").iterdir())
    (fold / "predictions.csv").write_text("unexpected\ncolumn\n", encoding="utf-8")
    with pytest.raises(Exception):
        validate_fold(fold, config_hash=config.config_hash, dataset_checksum="fixture")
    _, resumed = run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture", resume=True)
    assert resumed["completed_fold_count"] == 8
    assert any((root / "corrupt").iterdir())


def test_existing_completed_fold_refuses_overwrite_without_resume(tmp_path):
    dataset = _dataset(); config = parse_model_validation_config(_config())
    definition = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    root, _ = run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    prediction = next((root / "folds").glob("*/predictions.csv")); before = prediction.read_bytes()
    with pytest.raises(Exception, match="Completed fold exists"):
        run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    assert prediction.read_bytes() == before


def test_resume_rejects_provenance_mismatch_without_overwrite(tmp_path):
    import json
    dataset = _dataset(); config = parse_model_validation_config(_config())
    definition = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    root, _ = run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    metadata_path = next((root / "folds").glob("*/fold_metadata.json")); original = metadata_path.read_bytes()
    payload = json.loads(original); payload["config_hash"] = "mismatch"; metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="provenance mismatch"):
        run_folded_model_validation(config=config, dataset=dataset, nested_cv=definition, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture", resume=True)
    assert metadata_path.read_text(encoding="utf-8").find("mismatch") >= 0
