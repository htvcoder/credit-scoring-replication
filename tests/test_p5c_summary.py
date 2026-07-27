"""Summary reconciliation and interrupted-state acceptance tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import creditrep.experiments.model_validation as runner
from creditrep.artifacts.model_validation import reconcile_summary
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition


def _inputs():
    dataset = LoadedDataset("TOY", pd.DataFrame({"x": range(24)}), pd.Series([0, 1] * 12), {"numeric_columns": ["x"], "categorical_columns": [], "source_file": "fixture.csv", "row_count": 24, "feature_count": 1}, Path("fixture.csv"))
    config = parse_model_validation_config({"experiment": {"name": "summary_fixture", "publishable": False, "result_scope": "model_validation"}, "dataset": {"id": "TOY"}, "output": {"root_dir": "ignored"}, "models": {"logistic_regression": [{"max_iter": 30, "solver": "liblinear"}]}, "metrics": [{"id": "roc_auc"}], "optimization_metric": "roc_auc", "random_seed": 17})
    return dataset, config, create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)


def _run(tmp_path, **kwargs):
    dataset, config, nested = _inputs()
    return runner.run_folded_model_validation(config=config, dataset=dataset, nested_cv=nested, protocol_config=ProtocolAConfig(), output_root=tmp_path, dataset_checksum="fixture", **kwargs)


def test_resume_rebuilds_summary_without_duplicate_completed_units(tmp_path):
    root, first = _run(tmp_path)
    _, resumed = _run(tmp_path, resume=True)
    assert first["planned_fold_count"] == resumed["planned_fold_count"] == 2
    assert resumed["completed_fold_count"] == 2 and resumed["failed_fold_count"] == resumed["pending_fold_count"] == 0
    assert resumed["resumed_skipped_fold_count"] == 2
    assert len(list((root / "folds").iterdir())) == 2


def test_stale_summary_never_overrides_validated_fold_artifact_state(tmp_path):
    root, _ = _run(tmp_path)
    (root / "summary.json").write_text(json.dumps({"completed_fold_count": 999}), encoding="utf-8")
    _, summary = _run(tmp_path, resume=True)
    assert summary["completed_fold_count"] == 2 and summary["planned_fold_count"] == 2


def test_resume_handles_stale_temporary_fold_directory_safely(tmp_path):
    root, _ = _run(tmp_path)
    stale = root / "folds" / ".tmp-interrupted"; stale.mkdir(); (stale / "complete.json").write_text("{}", encoding="utf-8")
    _, summary = _run(tmp_path, resume=True)
    assert not stale.exists() and summary["completed_fold_count"] == 2


def test_artifact_write_failure_is_safely_retried_without_duplicate_outputs(tmp_path, monkeypatch):
    original = runner.write_fold_artifact; calls = {"count": 0}
    def fail_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1: raise OSError("write failure")
        return original(**kwargs)
    monkeypatch.setattr(runner, "write_fold_artifact", fail_once)
    root, first = _run(tmp_path)
    monkeypatch.setattr(runner, "write_fold_artifact", original)
    _, retried = _run(tmp_path, resume=True)
    assert first["failed_fold_count"] == 1
    assert retried["failed_fold_count"] == 0 and retried["completed_fold_count"] == 2
    assert retried["retried_fold_count"] == 1
    assert len(list((root / "folds").glob("*/predictions.csv"))) == 2


def test_summary_canonical_serialization_is_deterministic(tmp_path):
    root, _ = _run(tmp_path)
    _, config, nested = _inputs()
    units = {f"{outer.outer_fold_id}__logistic_regression": {"fold_hash": outer.split_hash, "model_id": "logistic_regression"} for outer in nested.outer_folds}
    first = reconcile_summary(experiment_root=root, planned_units=units, config=config, dataset_checksum="fixture")
    reversed_units = dict(reversed(list(units.items())))
    second = reconcile_summary(experiment_root=root, planned_units=reversed_units, config=config, dataset_checksum="fixture")
    for value in (first, second):
        value.pop("updated_at_utc")
    assert first == second
    assert second["planned_units"] == sorted(units)
    assert second["completed_units"] == sorted(units)


def test_reconciliation_replaces_stale_summary_with_canonical_artifact_state(tmp_path):
    root, _ = _run(tmp_path)
    stale = {"completed_fold_count": 999, "failed_fold_count": 999, "completed_units": ["fake", "fake"]}
    (root / "summary.json").write_text(json.dumps(stale), encoding="utf-8")
    _, summary = _run(tmp_path, resume=True)
    assert summary["completed_fold_count"] == 2
    assert summary["failed_fold_count"] == 0
    assert summary["completed_units"] == sorted(path.name for path in (root / "folds").iterdir())
