from __future__ import annotations

from copy import deepcopy

import pytest

from creditrep.experiments import p7c3_feasibility as harness


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
