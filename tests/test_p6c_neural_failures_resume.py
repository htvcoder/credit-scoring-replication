"""P6C.1C deterministic neural failure evidence tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import creditrep.experiments.model_validation as runner
from creditrep.artifacts.model_validation import (
    classify_retry,
    execution_unit_id,
    validate_failure_artifact,
    write_failure_artifact,
    reconcile_fold_state,
)
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition


def _context():
    return {
        "outer_fold_id": "repeat_00_fold_00",
        "inner_fold_id": "repeat_00_fold_00_inner_00",
        "candidate_id": 0,
        "training_scope": "inner_candidate",
        "model_seed": 17,
        "dataloader_seed": 17,
        "early_stopping_split_seed": 19,
        "config_fingerprint": "a" * 64,
        "execution_unit_id": execution_unit_id(
            experiment_id="exp",
            dataset_checksum="data",
            config_fingerprint="a" * 64,
            model_id="mlp_1",
            outer_fold_id="repeat_00_fold_00",
            inner_fold_id="repeat_00_fold_00_inner_00",
            candidate_id=0,
            training_scope="inner_candidate",
        ),
    }


def test_neural_failure_identity_sanitization_and_attempt_are_deterministic(tmp_path):
    context = _context()
    first = write_failure_artifact(
        root=tmp_path,
        experiment_id="exp",
        dataset_id="TOY",
        model_id="mlp_1",
        fold_id="repeat_00_fold_00__mlp_1",
        fold_hash="f" * 64,
        stage="neural_training",
        exception=OSError(
            "secret=abc token=xyz password=test D:\\private\\raw.csv raw-row=17 Tensor([1,2]) state_dict=weights"
        ),
        config_hash="a" * 64,
        neural_context=context,
    )
    one = validate_failure_artifact(first)
    second = write_failure_artifact(
        root=tmp_path,
        experiment_id="exp",
        dataset_id="TOY",
        model_id="mlp_1",
        fold_id="repeat_00_fold_00__mlp_1",
        fold_hash="f" * 64,
        stage="neural_training",
        exception=OSError("temporary lock"),
        config_hash="a" * 64,
        neural_context=context,
    )
    two = validate_failure_artifact(second)
    assert (
        one["execution_unit_id"]
        == two["execution_unit_id"]
        == context["execution_unit_id"]
    )
    assert two["attempt"] == two["attempt_number"] == 2
    assert (
        "abc" not in one["message"]
        and "private" not in one["message"]
        and "[PATH]" in one["message"]
    )
    assert "[1,2]" not in one["message"] and "weights" not in one["message"]
    assert two["retryable"] is True and two["retry_class"] == "retryable"


def test_neural_retry_classification_is_conservative():
    assert classify_retry(
        ValueError("non-finite training loss"), stage="neural_training"
    ) == ("non_retryable", False)
    assert classify_retry(
        OSError("temporary file lock"), stage="neural_artifact_publication"
    ) == ("retryable", True)
    assert classify_retry(
        ValueError("schema mismatch"), stage="neural_reconciliation"
    ) == ("non_retryable", False)


def test_execution_unit_changes_only_when_identity_changes():
    first = _context()["execution_unit_id"]
    base = {
        "experiment_id": "exp",
        "dataset_checksum": "data",
        "config_fingerprint": "a" * 64,
        "model_id": "mlp_1",
        "outer_fold_id": "repeat_00_fold_00",
        "inner_fold_id": "repeat_00_fold_00_inner_00",
        "candidate_id": 0,
        "training_scope": "inner_candidate",
    }
    assert execution_unit_id(**base) == first
    for field, changed_value in (
        ("outer_fold_id", "repeat_00_fold_01"),
        ("inner_fold_id", "repeat_00_fold_00_inner_01"),
        ("candidate_id", 1),
        ("training_scope", "final_refit"),
        ("config_fingerprint", "b" * 64),
    ):
        assert execution_unit_id(**(base | {field: changed_value})) != first


def _inputs():
    dataset = LoadedDataset(
        "TOY",
        pd.DataFrame({"x": range(24)}),
        pd.Series([0, 1] * 12),
        {
            "numeric_columns": ["x"],
            "categorical_columns": [],
            "source_file": "fixture.csv",
            "row_count": 24,
            "feature_count": 1,
        },
        Path("fixture.csv"),
    )
    config = parse_model_validation_config(
        {
            "experiment": {
                "name": "p6c_failure",
                "publishable": False,
                "result_scope": "model_validation",
            },
            "dataset": {"id": "TOY"},
            "output": {"root_dir": "ignored"},
            "models": {
                "mlp_1": [{"max_epochs": 1, "batch_size": 4, "device_policy": "cpu"}]
            },
            "metrics": [{"id": "roc_auc"}],
            "optimization_metric": "roc_auc",
            "random_seed": 17,
            "retry_policy": {"max_retry_attempts": 1},
        }
    )
    nested = create_nested_cv_definition(
        dataset,
        dataset_checksum="fixture",
        outer_n_repeats=1,
        inner_n_splits=2,
        random_seed=17,
    )
    return dataset, config, nested


def _run(tmp_path, **kwargs):
    dataset, config, nested = _inputs()
    return runner.run_folded_model_validation(
        config=config,
        dataset=dataset,
        nested_cv=nested,
        protocol_config=ProtocolAConfig(),
        output_root=tmp_path,
        dataset_checksum="fixture",
        **kwargs,
    )


@pytest.mark.parametrize(
    "point, stage",
    [
        ("split", "early_stopping_split"),
        ("initialization", "neural_model_initialization"),
        ("training", "neural_training"),
        ("capture", "neural_metadata_capture"),
    ],
)
def test_neural_operation_failures_keep_their_semantic_stage(
    tmp_path, monkeypatch, point, stage
):
    if point == "split":
        monkeypatch.setattr(
            runner,
            "create_early_stopping_split",
            lambda *a, **k: (_ for _ in ()).throw(
                ValueError("insufficient class samples")
            ),
        )
    elif point == "initialization":
        monkeypatch.setattr(
            runner,
            "create_model",
            lambda *a, **k: (_ for _ in ()).throw(OSError("temporary lock")),
        )
    elif point == "training":
        original = runner.create_model

        def fail_fit(*args, **kwargs):
            estimator = original(*args, **kwargs)
            estimator.fit = lambda *a, **k: (_ for _ in ()).throw(
                ValueError("non-finite training loss")
            )
            return estimator

        monkeypatch.setattr(runner, "create_model", fail_fit)
    else:
        monkeypatch.setattr(
            runner,
            "_neural_evidence",
            lambda **k: (_ for _ in ()).throw(ValueError("bad metadata")),
        )
    root, _ = _run(tmp_path)
    failure = validate_failure_artifact(next((root / "failures").glob("*.json")))
    assert failure["stage"] == stage
    assert failure["execution_unit_id"] and failure["model_seed"] >= 0


def test_neural_validation_and_publication_failures_are_atomic(tmp_path, monkeypatch):
    import creditrep.artifacts.model_validation as artifacts

    original = artifacts.validate_fold
    monkeypatch.setattr(
        artifacts,
        "validate_fold",
        lambda path, **k: (
            (_ for _ in ()).throw(artifacts.ArtifactError("synthetic validation"))
            if ".tmp-" in str(path)
            else original(path, **k)
        ),
    )
    root, summary = _run(tmp_path)
    assert summary["completed_fold_count"] == 0
    assert (
        validate_failure_artifact(next((root / "failures").glob("*.json")))["stage"]
        == "neural_artifact_validation"
    )
    assert not list((root / "folds").glob(".tmp-*"))


def test_neural_publication_failure_has_its_own_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "write_fold_artifact",
        lambda **kwargs: (_ for _ in ()).throw(OSError("temporary file lock")),
    )
    root, summary = _run(tmp_path)
    failure = validate_failure_artifact(next((root / "failures").glob("*.json")))
    assert summary["completed_fold_count"] == 0
    assert failure["stage"] == "neural_artifact_publication"
    assert failure["retryable"] is True and not list((root / "folds").glob(".tmp-*"))


def test_tiny_cpu_retry_preserves_identity_and_publishes(tmp_path, monkeypatch):
    original, calls = runner.create_model, {"count": 0}

    def transient(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("temporary lock")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "create_model", transient)
    root, first = _run(tmp_path)
    failure_path = next((root / "failures").glob("*.json"))
    before = validate_failure_artifact(failure_path)
    assert first["failed_fold_count"] == 1 and before["retryable"] is True
    monkeypatch.setattr(runner, "create_model", original)
    _, second = _run(tmp_path, resume=True)
    after = validate_failure_artifact(failure_path)
    assert second["completed_fold_count"] == 2 and after["resolved"] is True
    assert after["execution_unit_id"] == before["execution_unit_id"]
    assert (
        after["model_seed"] == before["model_seed"]
        and after["early_stopping_split_seed"] == before["early_stopping_split_seed"]
    )
    assert not list(root.rglob("*.pt")) and not list((root / "folds").glob(".tmp-*"))


def test_retry_budget_stops_after_the_configured_retry(tmp_path, monkeypatch):
    calls = {"count": 0}

    def always_transient(*args, **kwargs):
        calls["count"] += 1
        raise OSError("temporary lock")

    monkeypatch.setattr(runner, "create_model", always_transient)
    root, _ = _run(tmp_path)
    _, retry = _run(tmp_path, resume=True)
    count_after_budget = calls["count"]
    _, exhausted = _run(tmp_path, resume=True)
    assert retry["failed_fold_count"] == exhausted["failed_fold_count"] == 2
    assert calls["count"] == count_after_budget
    assert all(
        validate_failure_artifact(path)["attempt"] == 2
        for path in (root / "failures").glob("*.json")
    )


def test_corrupt_neural_fold_is_quarantined_then_reconciled(tmp_path):
    root, first = _run(tmp_path)
    fold = next((root / "folds").iterdir())
    (fold / "neural" / "final_refit" / "training_summary.json").unlink()
    _, resumed = _run(tmp_path, resume=True)
    failure = next(
        validate_failure_artifact(path)
        for path in (root / "failures").glob("*.json")
        if validate_failure_artifact(path)["stage"] == "neural_reconciliation"
    )
    assert first["completed_fold_count"] == 2 and resumed["completed_fold_count"] == 2
    assert failure["resolved"] is True and any((root / "corrupt").iterdir())


def test_reconciliation_distinguishes_completed_incomplete_corrupt_and_identity(
    tmp_path,
):
    root, _ = _run(tmp_path)
    dataset, config, nested = _inputs()
    outer = nested.outer_folds[0]
    fold_id = f"{outer.outer_fold_id}__mlp_1"
    arguments = {
        "root": root,
        "fold_id": fold_id,
        "config": config,
        "dataset_checksum": "fixture",
        "fold_hash": outer.split_hash,
        "model_id": "mlp_1",
    }
    assert reconcile_fold_state(**arguments) == "valid_completed"
    fold = root / "folds" / fold_id
    (fold / "neural" / "final_refit" / "training_history.json").unlink()
    assert reconcile_fold_state(**arguments) == "corrupt"
    (root / "folds" / ".tmp-missing").mkdir()
    assert reconcile_fold_state(**({**arguments, "fold_id": "missing"})) == "incomplete"
    wrong_config = parse_model_validation_config(
        {
            "experiment": {
                "name": "p6c_failure",
                "publishable": False,
                "result_scope": "model_validation",
            },
            "dataset": {"id": "TOY"},
            "output": {"root_dir": "ignored"},
            "models": {
                "mlp_1": [{"max_epochs": 2, "batch_size": 4, "device_policy": "cpu"}]
            },
            "metrics": [{"id": "roc_auc"}],
            "optimization_metric": "roc_auc",
            "random_seed": 17,
        }
    )
    wrong = {**arguments, "config": wrong_config}
    assert reconcile_fold_state(**wrong) == "identity_mismatch"
