from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditrep.datasets.models import LoadedDataset
from creditrep.experiments import p7c3_feasibility as harness
from creditrep.experiments.model_validation import _fit_for_partition
from creditrep.models import create_model
from creditrep.preprocessing import ProtocolAConfig
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
                            "parameters": {},
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
