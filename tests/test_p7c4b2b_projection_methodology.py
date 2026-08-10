import json
from pathlib import Path

import pytest

from creditrep.experiments.p7c4b2b_preflight import validate_artifacts
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2b import (
    MODES,
    PreflightError,
    build_plan,
    project,
    proposed_execution_plan,
    summarize,
)


MANIFEST = "configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml"


def _record(classification, mode, duration, start, **extra):
    return {
        "classification": classification,
        "mode": mode,
        "status": "completed",
        "wall_clock_seconds": duration,
        "started_monotonic": start,
        "completed_monotonic": start + duration,
        **extra,
    }


def _projection_records(mode_durations=(10.0, 15.0), sequential_mode_2=False):
    plan = build_plan(load_manifest(MANIFEST))
    records = []
    for mode, duration in zip(MODES, mode_durations):
        workers = MODES[mode]
        tick = 0.0
        batch = 0
        for task in plan["tasks"]:
            if task["mode"] != mode or task["classification"] != "measured":
                continue
            if mode == "cpu_parallel_2" and not sequential_mode_2:
                start = batch * duration
                if (
                    len([x for x in records if x["mode"] == mode]) % workers
                    == workers - 1
                ):
                    batch += 1
            else:
                start = tick
                tick += duration
            timing = _record("measured", mode, duration, start)
            records.append({**task, **timing})
    return records


def test_plan_separates_warmups_before_measured_phase():
    plan = build_plan(load_manifest(MANIFEST))
    for mode in MODES:
        classifications = [
            x["classification"] for x in plan["tasks"] if x["mode"] == mode
        ]
        assert classifications == ["warmup"] * 18 + ["measured"] * 36
    assert (
        plan["timing_scope_policy"]["execution_order"] == "all_warmups_before_measured"
    )


def test_warmup_duration_does_not_change_measured_projection_or_occupancy():
    measured = [
        _record("measured", "cpu_parallel_1", 10, 100),
        _record("measured", "cpu_parallel_1", 10, 110),
    ]
    short = summarize([_record("warmup", "cpu_parallel_1", 1, 0), *measured])
    long = summarize([_record("warmup", "cpu_parallel_1", 90, 0), *measured])
    for key in (
        "aggregate_measured_fit_runtime_seconds",
        "measured_phase_elapsed_seconds",
        "observed_worker_occupancy",
        "observed_elapsed_over_ideal_capacity_seconds",
    ):
        assert short[key] == long[key]
    assert short["warmup"]["aggregate_fit_runtime_seconds"] == 1
    assert long["warmup"]["aggregate_fit_runtime_seconds"] == 90


def test_zero_overhead_and_orchestration_gap_backchecks():
    perfect = summarize(
        [
            _record("measured", "cpu_parallel_2", 10, 0),
            _record("measured", "cpu_parallel_2", 10, 0),
        ],
        mode="cpu_parallel_2",
    )
    assert perfect["observed_worker_occupancy"] == pytest.approx(1)
    assert perfect["observed_elapsed_over_ideal_capacity_seconds"] == pytest.approx(0)
    assert perfect["bounded_workload_backcheck"][
        "reconstructed_elapsed_seconds"
    ] == pytest.approx(10)

    gap = summarize(
        [
            _record("measured", "cpu_parallel_1", 10, 0),
            _record("measured", "cpu_parallel_1", 10, 15),
        ]
    )
    assert gap["measured_idle_gap_seconds"] == pytest.approx(5)
    assert gap["observed_elapsed_over_ideal_capacity_seconds"] == pytest.approx(5)
    assert gap["bounded_workload_backcheck"][
        "reconstructed_elapsed_seconds"
    ] == pytest.approx(25)


def test_modes_share_formula_and_contention_is_not_penalized_twice():
    result = project(
        _projection_records(sequential_mode_2=True),
        evidence_scope="target_single_vm_measured",
    )
    p1 = result["single_vm_parallel_1"]
    p2 = result["single_vm_parallel_2"]
    for mode, value in (("cpu_parallel_1", p1), ("cpu_parallel_2", p2)):
        aggregate = value["inner_fit_projection"]["aggregate_fit_runtime_hours"][
            "point"
        ]
        elapsed = value["inner_fit_projection"][
            "conditional_work_conserving_elapsed_hours"
        ]
        assert elapsed["point"] == pytest.approx(aggregate / MODES[mode])
        assert elapsed["formula"] == "aggregate_fit_runtime_hours / workers"
    assert p2["measured_input"]["observed_worker_occupancy"] == pytest.approx(0.5)
    assert p2["inner_fit_projection"]["conditional_work_conserving_elapsed_hours"][
        "point"
    ] == pytest.approx(
        p1["inner_fit_projection"]["conditional_work_conserving_elapsed_hours"]["point"]
        * 0.75
    )


def test_projection_is_incomplete_warns_and_execution_plan_fails_closed():
    result = project(_projection_records(), evidence_scope="target_single_vm_measured")
    assert result["execution_plan_eligible"] is False
    assert "outer_refit_runtime_unmeasured" in result["execution_plan_blockers"]
    for mode in MODES:
        estimate = result[mode.replace("cpu_parallel", "single_vm_parallel")]
        assert estimate["warning"] == "high_extrapolation_ratio_non_guaranteed"
        assert estimate["outer_refits"]["status"] == "unknown_unmeasured"
        assert estimate["total_canonical_elapsed"]["projected_elapsed_hours"] is None
    with pytest.raises(PreflightError, match="projection_not_execution_plan_eligible"):
        proposed_execution_plan(
            git_commit="a" * 40,
            preflight_plan_digest="b" * 64,
            evidence_digest="c" * 64,
            mode="cpu_parallel_2",
            runtime_range=result["single_vm_parallel_2"],
            ram={},
            cost={},
        )


def test_incomplete_stratified_coverage_fails_closed():
    records = _projection_records()[:-1]
    result = project(records, evidence_scope="target_single_vm_measured")
    assert result["status"] in {
        "insufficient_stratified_coverage",
        "insufficient_stratified_repetitions",
    }


def test_real_run_regression_characteristics_do_not_reverse_mode_conclusion():
    p1_compute = 901.0254024187507
    p2_compute = 1243.270079388747
    old_p2_efficiency = 0.6352584167591891
    new_p1_elapsed = p1_compute
    new_p2_elapsed = p2_compute / 2
    old_p2_elapsed = p2_compute / (2 * old_p2_efficiency)
    assert 3012.788633243 / 2107.0468893880006 == pytest.approx(1.429857, rel=1e-5)
    assert new_p2_elapsed < new_p1_elapsed < old_p2_elapsed


@pytest.mark.parametrize(
    "run_name", ["cpu-parallel-1-attempt-001", "cpu-parallel-2-attempt-001"]
)
def test_supplied_legacy_artifacts_validate_and_project(run_name):
    run = Path("artifacts/p7c4b2b-compute-preflight") / run_name
    if not run.is_dir():
        pytest.skip("operator-supplied legacy artifact is not present")
    assert validate_artifacts(run)["valid"]
    records = [json.loads(path.read_text()) for path in run.glob("fits/*/result.json")]
    summary = summarize(
        records, mode=json.loads((run / "run_manifest.json").read_text())["mode"]
    )
    assert summary["measurement_phase_status"] == "interleaved_with_measured"
    assert summary["orchestration_overhead_status"].startswith("unknown_interleaved")
