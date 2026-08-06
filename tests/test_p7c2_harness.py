from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from creditrep.datasets.models import LoadedDataset
from creditrep.experiments import p7c2_feasibility as harness
from creditrep.experiments import p7c2_cli
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting import create_nested_cv_definition


def toy_dataset(dataset_id: str = "AC") -> LoadedDataset:
    rows = 60
    target = pd.Series([0, 1] * (rows // 2), name="BAD")
    features = pd.DataFrame(
        {
            "num": [float(index) for index in range(rows)],
            "cat": ["a" if index % 3 else "b" for index in range(rows)],
        }
    )
    return LoadedDataset(
        dataset_id=dataset_id,
        features=features,
        target=target,
        metadata={
            "dataset_id": dataset_id,
            "source_file": "synthetic/toy.csv",
            "target_column": "BAD",
            "removed_columns": [],
            "removed_identifier_columns": [],
            "removed_ignored_columns": [],
            "numeric_columns": ["num"],
            "categorical_columns": ["cat"],
        },
        source_path=Path("synthetic/toy.csv"),
    )


def execution_plan() -> dict:
    digest = "a" * 64
    fits = []
    for dataset_id in ("AC", "GMC"):
        for model_id, prefix in (("random_forest", "rf"), ("xgboost", "xgb")):
            for candidate_index in range(3):
                for inner_index in range(5):
                    identity = {
                        "plan_digest": digest,
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "outer_repeat_index": 0,
                        "outer_fold_index": 0,
                        "candidate_id": f"{prefix}_{candidate_index}",
                        "inner_fold_index": inner_index,
                        "seed": 100 + inner_index,
                    }
                    fit_id = harness.stable_fit_id(identity)
                    parameters = (
                        {"n_estimators": 1, "max_features_multiplier_of_sqrt_m": 1}
                        if model_id == "random_forest"
                        else {
                            "n_estimators": 1,
                            "max_depth": 1,
                            "learning_rate": 0.3,
                            "colsample_bytree": 0.6,
                            "subsample": 0.5,
                        }
                    )
                    fits.append(
                        {
                            **identity,
                            "fit_id": fit_id,
                            "parameters": parameters,
                            "dataset_checksum": "synthetic",
                            "inner_split_hash": f"split-{inner_index}",
                            "artifact_path": f"fits/{fit_id}/result.json",
                        }
                    )
    return {
        "schema_version": 1,
        "checkpoint_id": "P7C.2.1",
        "purpose": "engineering_feasibility_only",
        "plan_digest": digest,
        "threading": {
            "fits_parallelism": 1,
            "estimator_threads": 1,
            "allow_n_jobs_minus_one": False,
            "gpu_enabled": False,
        },
        "artifact_root": "artifacts/p7c2-rf-xgboost-feasibility",
        "retry_policy": {"max_retry_attempts": 1, "transient_errors_only": True},
        "fits": fits,
    }


def provenance() -> dict:
    return {
        "git_head": "b" * 40,
        "working_tree": "clean",
        "working_tree_details": {"is_dirty": False, "porcelain_v1": []},
    }


def completed_payload(fit: dict) -> dict:
    return {
        "schema_version": 1,
        "status": "completed",
        "outcome": "completed",
        "fit_id": fit["fit_id"],
        "plan_digest": fit["plan_digest"],
        "model_id": fit["model_id"],
        "dataset_id": fit["dataset_id"],
        "candidate_id": fit["candidate_id"],
        "outer_repeat_index": 0,
        "outer_fold_index": 0,
        "inner_fold_index": fit["inner_fold_index"],
        "seed": fit["seed"],
        "configured_thread_count": 1,
        "effective_thread_count": 1,
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "duration_seconds": 0.01,
        "process_cpu_seconds": 0.01,
        "process_id": 1,
        "process_rss_start_bytes": 1,
        "process_rss_peak_bytes": 2,
        "process_rss_delta_peak_bytes": 1,
        "library_versions": {},
        "git_provenance": provenance(),
        "error": None,
    }


def configure_synthetic_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        harness, "capture_git_provenance", lambda root, required: provenance()
    )
    monkeypatch.setattr(
        harness,
        "load_protocol_a_config",
        lambda repo_root: ProtocolAConfig(
            woe_enabled=True, vif_enabled=False, scaling_enabled=False
        ),
    )


def test_stable_fit_identity_changes_with_each_identity_axis():
    fit = execution_plan()["fits"][0]
    identity = {
        key: fit[key]
        for key in (
            "plan_digest",
            "model_id",
            "dataset_id",
            "outer_repeat_index",
            "outer_fold_index",
            "candidate_id",
            "inner_fold_index",
            "seed",
        )
    }
    baseline = harness.stable_fit_id(identity)
    for key, replacement in (
        ("plan_digest", "c" * 64),
        ("model_id", "xgboost"),
        ("dataset_id", "GMC"),
        ("outer_repeat_index", 1),
        ("outer_fold_index", 1),
        ("candidate_id", "other"),
        ("inner_fold_index", 4),
        ("seed", 999),
    ):
        changed = dict(identity)
        changed[key] = replacement
        assert harness.stable_fit_id(changed) != baseline


def test_execution_plan_is_exactly_sixty_and_rejects_thread_mutation():
    plan = execution_plan()
    assert harness.validate_execution_plan(plan)["unique_fit_ids"] == 60
    plan["threading"]["estimator_threads"] = -1
    with pytest.raises(harness.P7C2HarnessError, match="threading"):
        harness.validate_execution_plan(plan)


@pytest.mark.parametrize("model_id", ["random_forest", "xgboost"])
def test_two_tiny_synthetic_estimator_fits_use_real_preprocessing(model_id: str):
    dataset = toy_dataset()
    definition = create_nested_cv_definition(
        dataset,
        dataset_checksum="synthetic",
        outer_n_repeats=1,
        outer_n_splits=2,
        inner_n_splits=5,
        random_seed=42,
    )
    fit = next(
        item for item in execution_plan()["fits"] if item["model_id"] == model_id
    )
    inner = definition.outer_folds[0].inner_folds[0]
    fit["seed"] = inner.seed
    fit["fit_id"] = harness.stable_fit_id(
        {
            key: fit[key]
            for key in (
                "plan_digest",
                "model_id",
                "dataset_id",
                "outer_repeat_index",
                "outer_fold_index",
                "candidate_id",
                "inner_fold_index",
                "seed",
            )
        }
    )
    result = harness._execute_fit(
        fit,
        dataset,
        inner,
        ProtocolAConfig(woe_enabled=True, vif_enabled=False, scaling_enabled=False),
    )
    assert result["status"] == "completed"
    assert result["effective_thread_count"] == 1
    assert result["tree_method"] == ("hist" if model_id == "xgboost" else None)
    assert harness.FORBIDDEN_PAYLOAD_KEYS.isdisjoint(result)


@pytest.mark.parametrize("model_id", ["random_forest", "xgboost"])
def test_model_seed_parameters_threads_probability_and_fold_wiring(
    model_id: str, monkeypatch: pytest.MonkeyPatch
):
    fit = next(
        item for item in execution_plan()["fits"] if item["model_id"] == model_id
    )
    captured: dict = {}

    class Estimator:
        classes_ = np.asarray([0, 1])

        def fit(self, matrix, target):
            captured["train_rows"] = len(matrix)

        def predict_proba(self, matrix):
            return np.tile([0.25, 0.75], (len(matrix), 1))

    def preprocess(dataset, train_indices, transform_indices, protocol_config):
        captured["train_indices"] = train_indices
        captured["validation_indices"] = transform_indices
        return None, np.zeros((4, 9)), np.zeros((2, 9))

    def factory(received_model, parameters, random_seed):
        captured.update(
            model_id=received_model, parameters=parameters, seed=random_seed
        )
        return Estimator()

    monkeypatch.setattr(harness, "_fit_preprocessing", preprocess)
    monkeypatch.setattr(harness, "create_model", factory)
    inner = SimpleNamespace(train_indices=(0, 1, 2, 3), validation_indices=(4, 5))
    result = harness._execute_fit(
        fit,
        toy_dataset(),
        inner,
        ProtocolAConfig(woe_enabled=True, vif_enabled=False, scaling_enabled=False),
    )
    assert captured["train_indices"] == inner.train_indices
    assert captured["validation_indices"] == inner.validation_indices
    assert set(inner.train_indices).isdisjoint(inner.validation_indices)
    assert captured["model_id"] == model_id
    assert captured["seed"] == fit["seed"]
    assert captured["parameters"]["n_jobs"] == 1
    assert captured["parameters"]["random_state"] == fit["seed"]
    assert result["tree_method"] == ("hist" if model_id == "xgboost" else None)


def test_class_probability_contract_rejects_reversed_classes(
    monkeypatch: pytest.MonkeyPatch,
):
    class ReversedEstimator:
        classes_ = np.asarray([1, 0])

        def fit(self, matrix, target):
            return None

    monkeypatch.setattr(
        harness,
        "_fit_preprocessing",
        lambda *args, **kwargs: (None, np.zeros((4, 2)), np.zeros((2, 2))),
    )
    monkeypatch.setattr(
        harness, "create_model", lambda *args, **kwargs: ReversedEstimator()
    )
    fit = execution_plan()["fits"][0]
    with pytest.raises(harness.P7C2HarnessError, match="class order"):
        harness._execute_fit(
            fit,
            toy_dataset(),
            SimpleNamespace(train_indices=(0, 1, 2, 3), validation_indices=(4, 5)),
            ProtocolAConfig(woe_enabled=True, vif_enabled=False, scaling_enabled=False),
        )


def test_run_atomic_finalize_validate_and_resume_one_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    configure_synthetic_run(monkeypatch)
    plan = execution_plan()
    output = tmp_path / plan["artifact_root"] / "synthetic-run"
    calls: list[str] = []

    def execute(fit, dataset, inner, protocol):
        calls.append(fit["fit_id"])
        return completed_payload(fit)

    summary = harness.run(
        plan,
        output,
        repo_root=tmp_path,
        execute_fit=execute,
        dataset_loader=lambda dataset_id, repo_root: toy_dataset(dataset_id),
    )
    assert summary["completed"] == 60
    assert len(calls) == 60
    assert not list(output.rglob("*.tmp"))
    assert harness.validate_artifacts(plan, output)["completion_status"] == "completed"
    missing_fit = plan["fits"][17]
    (output / missing_fit["artifact_path"]).unlink()
    calls.clear()
    resumed = harness.run(
        plan,
        output,
        repo_root=tmp_path,
        resume=True,
        execute_fit=execute,
        dataset_loader=lambda dataset_id, repo_root: toy_dataset(dataset_id),
    )
    assert resumed["skipped"] == 59
    assert calls == [missing_fit["fit_id"]]


def test_failed_retry_budget_and_error_sanitization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    configure_synthetic_run(monkeypatch)
    plan = execution_plan()
    output = tmp_path / plan["artifact_root"] / "failed-run"
    target = plan["fits"][0]["fit_id"]

    def execute(fit, dataset, inner, protocol):
        if fit["fit_id"] == target:
            raise OSError(r"C:\private\secret.csv")
        return completed_payload(fit)

    first = harness.run(
        plan,
        output,
        repo_root=tmp_path,
        execute_fit=execute,
        dataset_loader=lambda dataset_id, repo_root: toy_dataset(dataset_id),
    )
    assert first["failed"] == 1
    failed_payload = json.loads(
        (output / plan["fits"][0]["artifact_path"]).read_text(encoding="utf-8")
    )
    assert failed_payload["error"]["message"] == "sanitized local-path error"
    second = harness.run(
        plan,
        output,
        repo_root=tmp_path,
        resume=True,
        execute_fit=execute,
        dataset_loader=lambda dataset_id, repo_root: toy_dataset(dataset_id),
    )
    assert second["failed"] == 1
    with pytest.raises(harness.P7C2HarnessError, match="resumable"):
        harness.run(
            plan,
            output,
            repo_root=tmp_path,
            resume=True,
            execute_fit=execute,
            dataset_loader=lambda dataset_id, repo_root: toy_dataset(dataset_id),
        )


def test_validator_detects_corrupt_unexpected_duplicate_and_temporary(tmp_path: Path):
    plan = execution_plan()
    output = tmp_path / "run"
    output.mkdir()
    harness._atomic_json(output / "execution_plan.json", plan)
    harness._atomic_json(
        output / "environment.json",
        {
            "schema_version": 1,
            "plan_digest": plan["plan_digest"],
            "threading": plan["threading"],
            **provenance(),
        },
    )
    fit = plan["fits"][0]
    path = output / fit["artifact_path"]
    harness._atomic_json(path, completed_payload(fit))
    harness._atomic_json(path.parent / "duplicate.json", completed_payload(fit))
    (path.parent / "bad.json").write_text("{", encoding="utf-8")
    (output / "orphan.tmp").write_text("partial", encoding="utf-8")
    report = harness.validate_artifacts(plan, output)
    codes = {item["code"] for item in report["errors"]}
    assert {"corrupt_json", "duplicate_fit", "incomplete_temporary"} <= codes
    unexpected = dict(completed_payload(fit), fit_id="f" * 64)
    harness._atomic_json(output / "fits" / ("f" * 64) / "result.json", unexpected)
    assert harness.validate_artifacts(plan, output)["valid"] is False


def test_validator_rejects_plan_threading_and_required_telemetry_mismatches(
    tmp_path: Path,
):
    plan = execution_plan()
    output = tmp_path / "run"
    output.mkdir()
    saved_plan = dict(plan, plan_digest="c" * 64)
    harness._atomic_json(output / "execution_plan.json", saved_plan)
    harness._atomic_json(
        output / "environment.json",
        {
            "schema_version": 1,
            "plan_digest": plan["plan_digest"],
            "threading": dict(plan["threading"], estimator_threads=2),
            **provenance(),
        },
    )
    fit = plan["fits"][0]
    payload = completed_payload(fit)
    del payload["process_id"]
    harness._atomic_json(output / fit["artifact_path"], payload)
    report = harness.validate_artifacts(plan, output)
    codes = {item["code"] for item in report["errors"]}
    assert {"plan_mismatch", "environment_mismatch", "invalid_telemetry"} <= codes
    assert report["valid"] is False


def test_run_rejects_invalid_plan_and_output_outside_artifact_root(tmp_path: Path):
    invalid = execution_plan()
    invalid["threading"]["fits_parallelism"] = 2
    with pytest.raises(harness.P7C2HarnessError, match="threading"):
        harness.run(invalid, tmp_path / "outside", repo_root=tmp_path)
    with pytest.raises(harness.P7C2HarnessError, match="child"):
        harness.run(execution_plan(), tmp_path / "outside", repo_root=tmp_path)


@pytest.mark.parametrize("command", ["plan", "validate-plan", "validate-artifacts"])
def test_read_only_cli_operations_never_call_run(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(p7c2_cli, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        p7c2_cli, "build_execution_plan", lambda *args, **kwargs: execution_plan()
    )
    monkeypatch.setattr(
        p7c2_cli,
        "run",
        lambda *args, **kwargs: pytest.fail("read-only CLI invoked training"),
    )
    monkeypatch.setattr(
        p7c2_cli,
        "validate_artifacts",
        lambda *args, **kwargs: {
            "valid": True,
            "expected": 60,
            "completion_status": "completed",
        },
    )
    argv = ["p7c2", command]
    if command == "validate-artifacts":
        argv += ["--output-dir", str(tmp_path / "artifacts")]
    monkeypatch.setattr(sys, "argv", argv)
    assert p7c2_cli.main() == 0


def test_validate_artifacts_cli_returns_nonzero_for_incomplete_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(p7c2_cli, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        p7c2_cli,
        "build_execution_plan",
        lambda *args, **kwargs: execution_plan(),
    )
    monkeypatch.setattr(
        p7c2_cli,
        "validate_artifacts",
        lambda *args, **kwargs: {
            "valid": True,
            "expected": 60,
            "completed": 59,
            "missing": 1,
            "completion_status": "incomplete",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["p7c2", "validate-artifacts", "--output-dir", str(tmp_path / "run")],
    )
    assert p7c2_cli.main() == 2


def test_interruption_writes_resumable_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    configure_synthetic_run(monkeypatch)
    plan = execution_plan()
    output = tmp_path / plan["artifact_root"] / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        harness.run(
            plan,
            output,
            repo_root=tmp_path,
            execute_fit=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
            dataset_loader=lambda dataset_id, repo_root: toy_dataset(dataset_id),
        )
    summary = json.loads(
        (output / "engineering_summary.json").read_text(encoding="utf-8")
    )
    assert summary["completion_status"] == "interrupted"
    assert harness.validate_artifacts(plan, output)["resumable"] is True
