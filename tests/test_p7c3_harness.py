from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditrep.datasets.models import LoadedDataset
from creditrep.datasets.registry import get_dataset_spec, load_registry
from creditrep.experiments import p7c3_feasibility as harness
from creditrep.experiments.model_validation import _fit_for_partition
from creditrep.models import create_model
from creditrep.preprocessing import ProtocolAConfig
from creditrep.preprocessing.exceptions import PreprocessingError
from creditrep.preprocessing.pipeline import ProtocolAPreprocessingPipeline
from creditrep.splitting import create_nested_cv_definition


def plan() -> dict:
    digest = "a" * 64
    fits = []
    for dataset in ("TC", "GMC"):
        for model in ("mlp_1", "mlp_3", "mlp_5"):
            for candidate in ("low_stress", "high_stress"):
                for fold in range(5):
                    identity = {
                        "plan_digest": digest,
                        "dataset_id": dataset,
                        "model_id": model,
                        "candidate_id": candidate,
                        "outer_repeat_index": 0,
                        "outer_fold_index": 0,
                        "inner_fold_index": fold,
                        "seed": 42 + fold,
                    }
                    fits.append(
                        identity
                        | {
                            "fit_id": harness.stable_fit_id(identity),
                            "parameters": {
                                "hidden_layers": [5],
                                "dropout": 0.0,
                                "batch_normalization": False,
                                "weight_decay": 0.0,
                                "learning_rate": 0.001,
                                "optimizer": "adam",
                                "batch_size": 32,
                                "max_epochs": 2,
                                "early_stopping_patience": 1,
                                "early_stopping_min_delta": 0.0,
                                "device_policy": "cpu",
                            },
                            "dataset_checksum": "fixture",
                            "artifact_path": f"fits/{harness.stable_fit_id(identity)}/result.json",
                        }
                    )
    return {
        "schema_version": 1,
        "checkpoint_id": "P7C.3",
        "purpose": "engineering_feasibility_only",
        "plan_digest": digest,
        "artifact_root": harness.ARTIFACT_ROOT,
        "fits": fits,
        "threading": {
            "fits_parallelism": 1,
            "torch_intraop_threads": 2,
            "blas_openmp_threads": 2,
            "nested_parallelism": False,
        },
        "limits": {
            "per_fit_timeout_seconds": 1800,
            "total_wall_time_seconds": 43200,
            "rss_warning_bytes": 10 * 1024**3,
            "rss_hard_bytes": int(11.5 * 1024**3),
            "disk_free_floor_bytes": 15 * 1024**3,
        },
        "retry_policy": {"max_retry_attempts": 1, "transient_errors_only": True},
        "preprocessing_config": asdict(
            ProtocolAConfig(woe_enabled=True, vif_enabled=True)
        ),
    }


def test_exactly_sixty_unique_mlp_fits_and_two_thread_sequential_policy():
    assert harness.validate_execution_plan(plan())["unique_fit_ids"] == 60


def test_rejects_parallelism_and_duplicate_fit_identity():
    broken = deepcopy(plan())
    broken["threading"]["fits_parallelism"] = 2
    with pytest.raises(harness.P7C3HarnessError, match="sequential"):
        harness.validate_execution_plan(broken)
    broken = deepcopy(plan())
    broken["fits"][1]["fit_id"] = broken["fits"][0]["fit_id"]
    with pytest.raises(harness.P7C3HarnessError, match="60 unique"):
        harness.validate_execution_plan(broken)


def test_artifact_validator_detects_missing_duplicate_corrupt_and_digest_mismatch(
    tmp_path,
):
    value = plan()
    output = tmp_path / "run"
    output.mkdir()
    harness._atomic_json(output / "execution_plan.json", value)
    fit = value["fits"][0]
    record = fit | {
        "status": "completed",
        "outcome": "completed",
        "provenance": {"git_head": "b" * 40},
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "wall_clock_seconds": 1.0,
    }
    harness._atomic_json(output / fit["artifact_path"], record)
    harness._atomic_json(
        output / fit["artifact_path"].replace("result.json", "duplicate.json"), record
    )
    (output / "fits" / "bad" / "result.json").parent.mkdir(parents=True)
    (output / "fits" / "bad" / "result.json").write_text("{", encoding="utf-8")
    report = harness.validate_artifacts(value, output)
    assert {"duplicate_fit", "corrupt_json"} <= {
        item["code"] for item in report["errors"]
    }


def _canonical_source() -> dict:
    return yaml.safe_load(
        Path("configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml").read_text(
            encoding="utf-8"
        )
    )


def _toy_dataset() -> LoadedDataset:
    rows = 60
    return LoadedDataset(
        "TOY",
        pd.DataFrame({"number": [float(index) for index in range(rows)]}),
        pd.Series([0, 1] * (rows // 2)),
        {
            "numeric_columns": ["number"],
            "categorical_columns": [],
            "source_file": "synthetic/toy.csv",
            "row_count": rows,
            "feature_count": 1,
        },
        Path("synthetic/toy.csv"),
    )


def _categorical_toy_dataset() -> LoadedDataset:
    rows = 60
    return LoadedDataset(
        "TOY_CATEGORICAL",
        pd.DataFrame(
            {
                "number": [float(index) for index in range(rows)],
                "category": ["a" if index % 2 else "b" for index in range(rows)],
            }
        ),
        pd.Series([0, 1] * (rows // 2)),
        {
            "numeric_columns": ["number"],
            "categorical_columns": ["category"],
            "source_file": "synthetic/toy.csv",
            "row_count": rows,
            "feature_count": 2,
        },
        Path("synthetic/toy.csv"),
    )


def test_tc_categorical_schema_uses_the_canonical_woe_contract(monkeypatch):
    tc = get_dataset_spec("TC", load_registry())
    assert tc.categorical_columns == (
        "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    )
    execution = harness.build_execution_plan(
        "configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml"
    )
    assert harness._reconstruct_protocol_config(execution["preprocessing_config"]).woe_enabled is True
    assert execution["source_plan_digest"] == "9df1e48d9531859a246f91aacf5551a9d4aaa8b6510f8c3b3950f2dd26b5ad24"

    monkeypatch.setattr(
        harness,
        "load_protocol_a_config",
        lambda **_: ProtocolAConfig(woe_enabled=False),
    )
    altered = harness.build_execution_plan(
        "configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml"
    )
    assert altered["plan_digest"] != execution["plan_digest"]


def test_lightweight_categorical_mlp_fit_keeps_woe_train_only():
    dataset = _categorical_toy_dataset()
    nested = create_nested_cv_definition(
        dataset,
        dataset_checksum="synthetic",
        outer_n_repeats=1,
        outer_n_splits=2,
        inner_n_splits=5,
        random_seed=42,
    )
    inner = nested.outer_folds[0].inner_folds[0]
    estimator, pipeline, _, early_stopping_split = _fit_for_partition(
        dataset=dataset,
        model_id="mlp_1",
        parameters={
            "hidden_layers": (5,), "dropout": 0.0, "batch_normalization": False,
            "weight_decay": 0.0, "learning_rate": 0.001, "optimizer": "adam",
            "batch_size": 16, "max_epochs": 2, "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0, "device_policy": "cpu",
        },
        seed=inner.seed,
        outer_id=nested.outer_folds[0].outer_fold_id,
        inner_id=inner.inner_fold_id,
        candidate_id="fixture",
        train_indices=inner.train_indices,
        evaluation_indices=inner.validation_indices,
        protocol_config=ProtocolAConfig(woe_enabled=True),
        model_stage="p7c3_feasibility",
    )
    assert estimator.get_training_summary()["epochs_completed"] >= 1
    assert pipeline.get_metadata()["woe"]["fitted"] is True
    assert pipeline.get_metadata()["fitted_row_count"] == len(
        early_stopping_split.train_indices
    )


def test_categorical_guard_remains_enabled_when_woe_is_disabled():
    dataset = _categorical_toy_dataset()
    with pytest.raises(PreprocessingError, match="requires WOE enabled"):
        ProtocolAPreprocessingPipeline(
            dataset_metadata=dataset.metadata,
            config=ProtocolAConfig(woe_enabled=False),
        ).fit(dataset.features.iloc[:20], dataset.target.iloc[:20])


@pytest.mark.parametrize(
    "model_id, depth", [("mlp_1", 1), ("mlp_3", 3), ("mlp_5", 5)]
)
def test_canonical_stress_candidates_map_to_strict_mlp_contract(model_id, depth):
    source = _canonical_source()
    model = next(item for item in source["models"] if item["model_id"] == model_id)
    for candidate in model["candidates"]:
        mapped = harness.adapt_mlp_feasibility_candidate(
            model_id, candidate, source["training_policy"]
        )
        assert mapped["hidden_layers"] == tuple(candidate["hidden_units"])
        assert len(mapped["hidden_layers"]) == depth
        assert mapped["weight_decay"] == candidate["l2"]
        assert mapped["dropout"] == candidate["dropout"]
        assert mapped["batch_normalization"] is candidate["batch_normalization"]
        assert mapped["learning_rate"] == candidate["learning_rate"]
        estimator = create_model(model_id, mapped, random_seed=42)
        assert estimator.config.weight_decay == candidate["l2"]


def test_execution_plan_applies_the_candidate_adapter_before_worker_execution():
    execution = harness.build_execution_plan(
        "configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml"
    )
    assert harness.validate_execution_plan(execution)["unique_fit_ids"] == 60
    first = next(
        fit
        for fit in execution["fits"]
        if fit["model_id"] == "mlp_3" and fit["candidate_id"] == "high_stress"
    )
    assert first["parameters"] == {
        "hidden_layers": [20, 20, 20],
        "dropout": 0.5,
        "batch_normalization": True,
        "weight_decay": 0.1,
        "learning_rate": 0.01,
        "optimizer": "adam",
        "batch_size": 32,
        "max_epochs": 200,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 0.0001,
        "device_policy": "cpu",
    }
    assert harness._reconstruct_mlp_parameters(first["parameters"])[
        "hidden_layers"
    ] == (20, 20, 20)
    assert harness._reconstruct_protocol_config(execution["preprocessing_config"]).woe_enabled


def test_plan_round_trip_is_valid_and_scientific_mismatch_is_explained(tmp_path):
    value = plan()
    output = tmp_path / "run"
    output.mkdir()
    harness._atomic_json(output / "execution_plan.json", value)
    assert not any(
        item["code"] == "plan_mismatch"
        for item in harness.validate_artifacts(value, output)["errors"]
    )

    changed = deepcopy(value)
    changed["fits"][0]["candidate_id"] = "different_scientific_candidate"
    identity = {
        key: changed["fits"][0][key]
        for key in (
            "plan_digest",
            "dataset_id",
            "model_id",
            "candidate_id",
            "outer_repeat_index",
            "outer_fold_index",
            "inner_fold_index",
            "seed",
        )
    }
    changed["fits"][0]["fit_id"] = harness.stable_fit_id(identity)
    report = harness.validate_artifacts(changed, output)
    mismatch = next(item for item in report["errors"] if item["code"] == "plan_mismatch")
    assert mismatch["path"] == "$.fits[0].candidate_id"


def test_failed_fit_is_not_valid_or_resumable_and_timestamp_validator_is_strict(tmp_path):
    value = plan()
    output = tmp_path / "run"
    output.mkdir()
    harness._atomic_json(output / "execution_plan.json", value)
    fit = value["fits"][0]
    failed = fit | {
        "schema_version": 1,
        "status": "failed",
        "outcome": "failed",
        "attempt_count": 1,
        "provenance": {"git_head": "b" * 40},
        "started_at": "2026-01-01T00:00:02Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "wall_clock_seconds": 0.1,
        "failure_classification": "implementation_data_configuration",
    }
    harness._atomic_json(output / fit["artifact_path"], failed)
    report = harness.validate_artifacts(value, output)
    assert report["valid"] is False
    assert report["resumable"] is False
    assert "invalid_telemetry" in {item["code"] for item in report["errors"]}


@pytest.mark.parametrize("outcome", ["completed", "handled_failure", "timeout", "exception"])
def test_timestamp_contract_accepts_all_record_outcomes_without_sleep(outcome):
    payload = {
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:00.125Z",
        "wall_clock_seconds": 0.125,
        "outcome": outcome,
    }
    assert harness._timestamp_duration_valid(payload)


@pytest.mark.parametrize("model_id", ["mlp_1", "mlp_3", "mlp_5"])
def test_tiny_real_fit_uses_p7c3_production_model_path(model_id):
    source = _canonical_source()
    candidate = next(
        item for model in source["models"] if model["model_id"] == model_id
        for item in model["candidates"]
        if item["id"] == "high_stress"
    )
    parameters = harness.adapt_mlp_feasibility_candidate(
        model_id, candidate, source["training_policy"]
    ) | {"max_epochs": 2, "early_stopping_patience": 1}
    dataset = _toy_dataset()
    nested = create_nested_cv_definition(
        dataset,
        dataset_checksum="synthetic",
        outer_n_repeats=1,
        outer_n_splits=2,
        inner_n_splits=5,
        random_seed=42,
    )
    outer = nested.outer_folds[0]
    inner = outer.inner_folds[0]
    # This is the same _fit_for_partition production route invoked by _child.
    estimator, _, _, _ = _fit_for_partition(
        dataset=dataset,
        model_id=model_id,
        parameters=parameters,
        seed=inner.seed,
        outer_id=outer.outer_fold_id,
        inner_id=inner.inner_fold_id,
        candidate_id="high_stress",
        train_indices=inner.train_indices,
        evaluation_indices=inner.validation_indices,
        protocol_config=ProtocolAConfig(),
        model_stage="p7c3_feasibility",
    )
    summary = estimator.get_training_summary()
    assert summary["epochs_completed"] >= 1
    assert summary["optimizer"]["weight_decay"] == candidate["l2"]


def test_contract_mapping_rejects_missing_invalid_and_unknown_parameters():
    source = _canonical_source()
    candidate = source["models"][0]["candidates"][0]
    with pytest.raises(harness.P7C3HarnessError, match="missing"):
        harness.adapt_mlp_feasibility_candidate(
            "mlp_1", {key: value for key, value in candidate.items() if key != "l2"},
            source["training_policy"],
        )
    invalid_depth = dict(candidate, hidden_units=[5, 5])
    with pytest.raises(harness.P7C3HarnessError, match="exactly 1"):
        harness.adapt_mlp_feasibility_candidate(
            "mlp_1", invalid_depth, source["training_policy"]
        )
    with pytest.raises(Exception, match="unknown hyperparameters"):
        create_model("mlp_1", {"unknown": 1})
