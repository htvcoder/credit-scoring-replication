"""Candidate failure and deterministic-selection acceptance tests for P5C."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import creditrep.experiments.model_validation as runner
from creditrep.artifacts.model_validation import validate_failure_artifact
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition


def _inputs():
    dataset = LoadedDataset("TOY", pd.DataFrame({"x": range(24)}), pd.Series([0, 1] * 12), {"numeric_columns": ["x"], "categorical_columns": [], "source_file": "fixture.csv", "row_count": 24, "feature_count": 1}, Path("fixture.csv"))
    payload = {"experiment": {"name": "candidate_fixture", "publishable": False, "result_scope": "model_validation"}, "dataset": {"id": "TOY"}, "output": {"root_dir": "ignored"}, "models": {"logistic_regression": [{"C": 0.1, "max_iter": 30, "solver": "liblinear"}, {"C": 1.0, "max_iter": 30, "solver": "liblinear"}]}, "metrics": [{"id": "roc_auc"}], "optimization_metric": "roc_auc", "random_seed": 17}
    config = parse_model_validation_config(payload)
    nested = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    return dataset, config, nested


def test_all_candidate_failures_produce_inner_tuning_failure_without_outer_refit(tmp_path, monkeypatch):
    dataset, config, nested = _inputs()
    calls = {"create": 0}
    def fail(*args, **kwargs):
        calls["create"] += 1
        raise ValueError("candidate failed")
    monkeypatch.setattr(runner, "create_model", fail)
    root, summary = runner.run_folded_model_validation(config=config, dataset=dataset, nested_cv=nested, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    failures = [validate_failure_artifact(path) for path in sorted((root / "failures").glob("*.json"))]
    assert calls["create"] == len(config.model_candidates["logistic_regression"]) * len(nested.outer_folds)
    assert failures and all(item["stage"] == "inner_tuning" for item in failures)
    assert summary["failed_fold_count"] == 2 and summary["completed_fold_count"] == 0
    assert not list((root / "folds").glob("*/complete.json"))


def test_invalid_candidate_is_excluded_when_valid_candidate_remains(monkeypatch):
    dataset, config, nested = _inputs()
    original = runner.create_model
    def selective(model_id, parameters, **kwargs):
        if parameters["C"] == 0.1:
            raise ValueError("invalid candidate")
        return original(model_id, parameters, **kwargs)
    monkeypatch.setattr(runner, "create_model", selective)
    result = runner._run_fold(config=config, dataset=dataset, outer=nested.outer_folds[0], model_id="logistic_regression", candidates=config.model_candidates["logistic_regression"], protocol_config=ProtocolAConfig())
    assert result["selected_candidate"]["parameters"]["C"] == 1.0
    assert len(result["warnings"]) == 1


def test_candidate_tie_breaking_is_deterministic_independent_of_input_order(monkeypatch):
    dataset, config, nested = _inputs()
    monkeypatch.setattr(runner, "_score", lambda *args, **kwargs: 0.5)
    candidates = config.model_candidates["logistic_regression"]
    first = runner._run_fold(config=config, dataset=dataset, outer=nested.outer_folds[0], model_id="logistic_regression", candidates=candidates, protocol_config=ProtocolAConfig())
    second = runner._run_fold(config=config, dataset=dataset, outer=nested.outer_folds[0], model_id="logistic_regression", candidates=tuple(reversed(candidates)), protocol_config=ProtocolAConfig())
    assert first["selected_candidate"]["parameters"] == second["selected_candidate"]["parameters"]
    assert first["selected_candidate"]["candidate_hash"] == second["selected_candidate"]["candidate_hash"]


def test_all_candidates_with_invalid_scores_fail_before_outer_refit(tmp_path, monkeypatch):
    dataset, config, nested = _inputs()
    monkeypatch.setattr(runner, "_score", lambda *args, **kwargs: float("nan"))
    root, summary = runner.run_folded_model_validation(config=config, dataset=dataset, nested_cv=nested, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture")
    failures = [validate_failure_artifact(path) for path in sorted((root / "failures").glob("*.json"))]
    assert failures and all(item["stage"] == "inner_tuning" for item in failures)
    assert summary["completed_fold_count"] == 0
    assert not list((root / "folds").glob("*/predictions.csv"))
