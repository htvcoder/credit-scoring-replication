"""Runtime failure and retry acceptance tests for P5C."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import creditrep.experiments.model_validation as runner
from creditrep.artifacts.model_validation import validate_failure_artifact
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition


def _inputs():
    dataset = LoadedDataset("TOY", pd.DataFrame({"x": range(24)}), pd.Series([0, 1] * 12), {"numeric_columns": ["x"], "categorical_columns": [], "source_file": "fixture.csv", "row_count": 24, "feature_count": 1}, Path("fixture.csv"))
    config = parse_model_validation_config({"experiment": {"name": "failure_fixture", "publishable": False, "result_scope": "model_validation"}, "dataset": {"id": "TOY"}, "output": {"root_dir": "ignored"}, "models": {"logistic_regression": [{"max_iter": 30, "solver": "liblinear"}]}, "metrics": [{"id": "roc_auc"}], "optimization_metric": "roc_auc", "random_seed": 17})
    nested = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    return dataset, config, nested


def _run(tmp_path, **kwargs):
    dataset, config, nested = _inputs()
    return runner.run_folded_model_validation(config=config, dataset=dataset, nested_cv=nested, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture", **kwargs)


def test_preprocessing_failure_is_recorded_and_continue_mode_runs_remaining_units(tmp_path, monkeypatch):
    original = runner._fit_preprocessing
    calls = {"count": 0}
    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("secret=abc\\raw-row")
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "_fit_preprocessing", fail_once)
    root, summary = _run(tmp_path)
    failure = validate_failure_artifact(next((root / "failures").glob("*.json")))
    assert failure["stage"] == "preprocessing"
    assert failure["exception_type"] == "ValueError"
    assert failure["message"] == "secret=[REDACTED]" and "\n" not in failure["message"]
    assert summary["failed_fold_count"] == 1
    assert summary["completed_fold_count"] == 1


def test_fail_fast_records_first_failure_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_fit_preprocessing", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(RuntimeError, match="Fold failed"):
        _run(tmp_path, fail_fast=True)
    failures = list(tmp_path.rglob("failures/*.json"))
    assert len(failures) == 1


def test_failed_fold_retries_and_resolves_failure_evidence(tmp_path, monkeypatch):
    original = runner._run_fold
    calls = {"count": 0}
    def fail_first(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("first failure")
        return original(**kwargs)
    monkeypatch.setattr(runner, "_run_fold", fail_first)
    root, first = _run(tmp_path)
    assert first["failed_fold_count"] == 1
    monkeypatch.setattr(runner, "_run_fold", original)
    _, retried = _run(tmp_path, resume=True)
    failure = validate_failure_artifact(next((root / "failures").glob("*.json")))
    assert failure["resolved"] is True
    assert failure["attempt"] == 1
    assert retried["failed_fold_count"] == 0
    assert retried["completed_fold_count"] == 2
    assert retried["retried_fold_count"] == 1


@pytest.mark.parametrize("stage, target", [("inner_tuning", "inner"), ("outer_refit", "outer"), ("prediction", "prediction"), ("metrics", "metrics")])
def test_runtime_failures_are_classified_at_their_real_stage(tmp_path, monkeypatch, stage, target):
    if target == "inner":
        monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("inner boom")))
    elif target == "outer":
        original, calls = runner.create_model, {"count": 0}
        def fail_outer(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise ValueError("outer boom")
            return original(*args, **kwargs)
        monkeypatch.setattr(runner, "create_model", fail_outer)
    elif target == "prediction":
        original, calls = runner.positive_class_probabilities, {"count": 0}
        def fail_prediction(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise ValueError("prediction boom")
            return original(*args, **kwargs)
        monkeypatch.setattr(runner, "positive_class_probabilities", fail_prediction)
    else:
        original, calls = runner.compute_configured_metric, {"count": 0}
        def fail_metrics(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise ValueError("metrics boom")
            return original(*args, **kwargs)
        monkeypatch.setattr(runner, "compute_configured_metric", fail_metrics)
    root, summary = _run(tmp_path)
    failure = validate_failure_artifact(next((root / "failures").glob("*.json")))
    assert failure["stage"] == stage
    assert summary["failed_fold_count"] >= 1


def test_artifact_write_and_validation_failures_never_promote_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "write_fold_artifact", lambda **kwargs: (_ for _ in ()).throw(OSError("disk error")))
    root, summary = _run(tmp_path)
    assert summary["completed_fold_count"] == 0
    assert validate_failure_artifact(next((root / "failures").glob("*.json")))["stage"] == "artifact_write"
    assert not list((root / "folds").iterdir())


def test_artifact_validation_failure_is_recorded_and_not_promoted(tmp_path, monkeypatch):
    import creditrep.artifacts.model_validation as artifacts
    original = artifacts.validate_fold
    def reject_temporary(path, **kwargs):
        if ".tmp-" in str(path):
            raise artifacts.ArtifactError("synthetic validation rejection")
        return original(path, **kwargs)
    monkeypatch.setattr(artifacts, "validate_fold", reject_temporary)
    root, summary = _run(tmp_path)
    assert summary["completed_fold_count"] == 0
    assert validate_failure_artifact(next((root / "failures").glob("*.json")))["stage"] == "artifact_validation"
    assert not list((root / "folds").iterdir())
