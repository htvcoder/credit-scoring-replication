from copy import deepcopy
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from creditrep.config.loader import sha256_canonical
from creditrep.experiments import p7c4b2b_preflight as runner
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2b import (
    DATASET_FINGERPRINTS,
    SCIENTIFIC_DIGEST,
    PreflightError,
    build_plan,
    cost_estimate,
    execution_approval_guard,
    machine_profile_digest,
    plan_digest,
    project,
    proposed_execution_plan,
    ram_feasibility,
    summarize,
    validate_execution_plan,
    validate_machine,
    validate_plan,
)

MANIFEST = "configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml"


def manifest():
    return load_manifest(MANIFEST)


def plan():
    return build_plan(manifest())


def profile(role="development_calibration_only"):
    value = {
        "schema_version": 1,
        "machine_role": role,
        "machine_id": "fixture-machine",
        "cloud_provider": "fixture-provider",
        "instance_type": "fixture-type",
        "os": "fixture",
        "cpu_model": "fixture",
        "physical_cores": 2,
        "logical_cores": 4,
        "ram_total_bytes": 32 * 1024**3,
        "disk_free_bytes": 32 * 1024**3,
        "python_executable": "python.exe",
        "python_version": "3.11",
        "dependency_fingerprint": "a" * 64,
        "git_commit": "b" * 40,
        "scientific_manifest_digest": SCIENTIFIC_DIGEST,
        "dataset_fingerprints": DATASET_FINGERPRINTS,
        "worker_limit": 2,
        "threads_per_worker": 2,
        "virtualization_power": "unknown",
        "utc_captured": "2026-08-09T00:00:00Z",
    }
    value["profile_digest"] = machine_profile_digest(value)
    return value


def root(tmp_path):
    value = tmp_path / "repo"
    (value / runner.ARTIFACT_ROOT).mkdir(parents=True)
    return value


def test_plan_deterministic_exact_budget_candidates_and_digest():
    a, b = plan(), plan()
    assert a == b
    report = validate_plan(a)
    assert (
        report["units"] == 18
        and report["warmups_per_mode"] == 18
        and report["measured_fits_per_mode"] == 36
    )
    assert a["fit_budget"] == {
        "warmups_per_mode": 18,
        "measured_per_mode": 36,
        "no_retry_total": 108,
        "retry_maximum_per_task": 1,
        "worst_case_attempts": 216,
    }
    ids = {
        item["candidate_id"]
        for model in manifest()["models"]
        for item in model["candidates"]
    }
    assert {unit["candidate_id"] for unit in a["units"]} <= ids
    assert {unit["coverage_role"] for unit in a["units"]} == {
        "light",
        "median",
        "heavy",
    }
    changed = deepcopy(a)
    changed["limits"]["global_wall_clock_seconds"]["value"] += 1
    with pytest.raises(PreflightError, match="plan_digest_mismatch"):
        validate_plan(changed)


def test_machine_target_gate_resource_compatibility_and_digest():
    target = profile("intended_single_vm_target")
    assert validate_machine(target, plan())["valid"]
    for mutation, match in (
        (
            lambda x: x.update(machine_role="development_calibration_only"),
            "target_machine_not_confirmed",
        ),
        (lambda x: x.update(ram_total_bytes=1), "target_ram_incompatible"),
    ):
        bad = deepcopy(target)
        mutation(bad)
        bad["profile_digest"] = machine_profile_digest(bad)
        with pytest.raises(PreflightError, match=match):
            validate_machine(bad, plan())


def test_summary_projection_ram_cost_and_execution_plan_fail_closed():
    records = [
        {"classification": "warmup", "mode": "cpu_parallel_1", "wall_clock_seconds": 1},
        {
            "classification": "measured",
            "mode": "cpu_parallel_1",
            "wall_clock_seconds": 2,
        },
        {
            "classification": "measured",
            "mode": "cpu_parallel_1",
            "wall_clock_seconds": 4,
        },
    ]
    summary = summarize(records)
    assert (
        summary["mean_seconds"] == 3
        and summary["warmups_excluded"] == 1
        and summary["p95_seconds"] == "insufficient_sample"
    )
    pending = project(records, evidence_scope=runner.EVIDENCE_FIXTURE)
    assert (
        pending["status"] == "pending_target_measurement"
        and pending["gpu"]["status"] == "pending_gpu_preflight"
    )
    assert pending["two_vm_cpu"]["status"].endswith("not_authorized")
    assert (
        ram_feasibility(records, profile(), evidence_scope=runner.EVIDENCE_FIXTURE)[
            "two_workers"
        ]
        == "memory_feasibility_uncertain"
    )
    assert cost_estimate(None, None)["status"] == "pending_operator_price_input"
    proposed = proposed_execution_plan(
        git_commit="b" * 40,
        preflight_plan_digest=plan()["plan_digest"],
        evidence_digest="e" * 64,
        mode="cpu_parallel_2",
        runtime_range="pending",
        ram="pending",
        cost="pending",
    )
    assert (
        proposed["approval"]["status"] == "pending_human_approval"
        and proposed["approval"]["approver"] is None
    )
    assert proposed["execution_plan_digest"] == sha256_canonical(
        {k: v for k, v in proposed.items() if k != "execution_plan_digest"}
    )
    assert (
        validate_execution_plan(proposed)["approval_status"] == "pending_human_approval"
    )
    assert execution_approval_guard(proposed, None)["reason_codes"] == [
        "execution_cost_approval_missing"
    ]
    approval = {
        "status": "approved",
        "execution_plan_digest": proposed["execution_plan_digest"],
        "approver": "human",
        "approved_utc": "2026-08-09T00:00:00Z",
    }
    assert execution_approval_guard(proposed, approval)["authorized"]
    changed = deepcopy(proposed)
    changed["runtime_range"] = "changed"
    assert (
        "execution_plan_digest_mismatch"
        in execution_approval_guard(changed, approval)["reason_codes"]
    )


def test_fixture_runner_atomic_resume_threads_and_artifact_validation(tmp_path):
    repo = root(tmp_path)
    output = repo / runner.ARTIFACT_ROOT / "fixture-ok"
    p = plan()
    prof = profile()
    first = runner.run(
        p,
        prof,
        output,
        mode="cpu_parallel_2",
        repo_root=repo,
        fixture=True,
        max_tasks=3,
        timeout_seconds=5,
    )
    assert first["validation"]["valid"] and first["completed"] == 3
    records = [
        json.loads(path.read_text()) for path in output.glob("fits/*/result.json")
    ]
    assert all(
        x["max_workers"] <= 2
        and x["threads_per_worker"] == 2
        and x["worker_identity"] != os.getpid()
        for x in records
    )
    assert all(set(x["thread_env"].values()) <= {"2", "false"} for x in records)
    assert not list(output.rglob("*.tmp")) and (output / "COMPLETED.json").is_file()
    second = runner.resume(
        p,
        prof,
        output,
        mode="cpu_parallel_2",
        repo_root=repo,
        fixture=True,
        max_tasks=3,
        timeout_seconds=5,
    )
    assert second["skipped"] == 3 and second["executed"] == 0


def test_timeout_crash_retry_and_corrupt_resume_are_fail_closed(tmp_path):
    repo = root(tmp_path)
    prof = profile()
    sleeping = plan()
    sleeping["tasks"][0].update(fixture_behavior="sleep", fixture_sleep_seconds=2)
    sleeping["plan_digest"] = plan_digest(sleeping)
    timed = repo / runner.ARTIFACT_ROOT / "fixture-timeout"
    result = runner.run(
        sleeping,
        prof,
        timed,
        mode="cpu_parallel_1",
        repo_root=repo,
        fixture=True,
        max_tasks=1,
        timeout_seconds=0.1,
    )
    assert result["failed"] == 1 and not (timed / "COMPLETED.json").exists()
    transient = plan()
    transient["tasks"][0]["fixture_behavior"] = "transient"
    transient["plan_digest"] = plan_digest(transient)
    retried = repo / runner.ARTIFACT_ROOT / "fixture-retry"
    result = runner.run(
        transient,
        prof,
        retried,
        mode="cpu_parallel_1",
        repo_root=repo,
        fixture=True,
        max_tasks=1,
        timeout_seconds=10,
    )
    attempts = list(retried.glob("fits/*/attempts/attempt-*/telemetry.json"))
    assert result["failed"] == 1 and len(attempts) == 2
    ok = repo / runner.ARTIFACT_ROOT / "fixture-corrupt"
    runner.run(
        plan(),
        prof,
        ok,
        mode="cpu_parallel_1",
        repo_root=repo,
        fixture=True,
        max_tasks=3,
        timeout_seconds=10,
    )
    path = next(ok.glob("fits/*/result.json"))
    path.write_text("{bad", encoding="utf-8")
    assert "corrupt_json" in runner.validate_artifacts(ok)["reason_codes"]
    with pytest.raises(PreflightError, match="corrupt_completed_artifact"):
        runner.resume(
            plan(),
            prof,
            ok,
            mode="cpu_parallel_1",
            repo_root=repo,
            fixture=True,
            max_tasks=3,
        )


@pytest.mark.parametrize(
    "mutation,code",
    [
        (
            lambda run: (run / "plan.json").write_text("{}", encoding="utf-8"),
            "plan_digest_mismatch",
        ),
        (
            lambda run: (run / "machine_profile.json").write_text(
                "{}", encoding="utf-8"
            ),
            "machine_profile_digest_mismatch",
        ),
        (
            lambda run: next(run.glob("fits/*/result.json")).write_text(
                "[]", encoding="utf-8"
            ),
            "corrupt_json",
        ),
    ],
)
def test_validator_stable_mutation_codes(tmp_path, mutation, code):
    repo = root(tmp_path)
    output = repo / runner.ARTIFACT_ROOT / "fixture-mutation"
    runner.run(
        plan(),
        profile(),
        output,
        mode="cpu_parallel_1",
        repo_root=repo,
        fixture=True,
        max_tasks=3,
        timeout_seconds=10,
    )
    mutation(output)
    assert code in runner.validate_artifacts(output)["reason_codes"]


def test_cli_help_plan_and_development_real_run_guard():
    repo = Path(__file__).resolve().parents[1]
    base = [sys.executable, "-m", "creditrep.experiments.p7c4b2b_cli"]
    help_result = subprocess.run(
        [*base, "--help"], cwd=repo, capture_output=True, text=True, timeout=30
    )
    assert help_result.returncode == 0
    validated = subprocess.run(
        [*base, "validate-plan"], cwd=repo, capture_output=True, text=True, timeout=30
    )
    assert validated.returncode == 0
    output = f"artifacts/p7c4b2b-compute-preflight/guard-{uuid.uuid4().hex[:8]}"
    guarded = subprocess.run(
        [*base, "run", "--output-dir", output],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        guarded.returncode == runner.EXIT_GUARD
        and "target_machine_assertion_missing" in guarded.stdout
    )


def test_validator_detects_identity_telemetry_summary_completion_and_partial_mutations(
    tmp_path,
):
    repo = root(tmp_path)
    baseline = repo / runner.ARTIFACT_ROOT / "fixture-baseline"
    runner.run(
        plan(),
        profile(),
        baseline,
        mode="cpu_parallel_1",
        repo_root=repo,
        fixture=True,
        max_tasks=3,
        timeout_seconds=10,
    )
    def case(name, mutate, code):
        target = repo / runner.ARTIFACT_ROOT / name
        shutil.copytree(baseline, target)
        manifest = json.loads((target / "run_manifest.json").read_text())
        manifest["run_id"] = target.name
        (target / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        completion = json.loads((target / "COMPLETED.json").read_text())
        completion["run_id"] = target.name
        (target / "COMPLETED.json").write_text(json.dumps(completion), encoding="utf-8")
        for path in target.glob("fits/*/result.json"):
            value = json.loads(path.read_text())
            value["run_id"] = target.name
            path.write_text(json.dumps(value), encoding="utf-8")
        mutate(target)
        assert code in runner.validate_artifacts(target)["reason_codes"]

    case(
        "bad-timing",
        lambda x: _mutate_json(
            next(x.glob("fits/*/result.json")), "wall_clock_seconds", -1
        ),
        "invalid_timing",
    )
    case(
        "bad-memory",
        lambda x: _mutate_json(
            next(x.glob("fits/*/result.json")), "process_peak_rss_bytes", -1
        ),
        "invalid_memory",
    )
    case(
        "bad-thread",
        lambda x: _mutate_json(
            next(x.glob("fits/*/result.json")), "threads_per_worker", 3
        ),
        "thread_limit_violation",
    )
    case(
        "bad-identity",
        lambda x: _mutate_json(
            next(x.glob("fits/*/result.json")), "candidate_id", "foreign"
        ),
        "retry_identity_mismatch",
    )
    case(
        "bad-summary",
        lambda x: _mutate_json(x / "summary.json", "successful", 999),
        "summary_mismatch",
    )
    case(
        "partial",
        lambda x: (x / "temporary" / "foreign.tmp").mkdir(parents=True),
        "partial_temporary_attempt",
    )
    case(
        "forged-marker",
        lambda x: _mutate_json(x / "COMPLETED.json", "validation_report_digest", "bad"),
        "premature_completion_marker",
    )


def _mutate_json(path, key, value):
    payload = json.loads(path.read_text())
    payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stratified_projection_requires_complete_target_evidence_and_nonperfect_two_vm_efficiency():
    records = []
    tick = 0.0
    for task in plan()["tasks"]:
        if task["classification"] != "measured":
            continue
        duration = (
            1
            + {"mlp_1": 1, "mlp_3": 2, "mlp_5": 3}[task["model_id"]]
            + {"light": 0, "median": 1, "heavy": 2}[task["coverage_role"]]
            + (3 if task["dataset_id"] == "GMC" else 0)
        )
        records.append(
            {
                **task,
                "status": "completed",
                "wall_clock_seconds": duration,
                "started_monotonic": tick,
                "completed_monotonic": tick + duration,
                "aggregate_process_tree_peak_rss_bytes": 100,
                "system_available_ram_min_bytes": 8 * 1024**3,
            }
        )
        tick += duration
    derived = project(
        records, evidence_scope="target_single_vm_measured", two_vm_efficiency=0.8
    )
    assert (
        derived["status"].startswith("derived")
        and derived["single_vm_parallel_2"]["derived_value"]["projected_elapsed_hours"]
        > 0
    )
    assert (
        derived["two_vm_cpu"]["efficiency"] == 0.8
        and derived["gpu"]["status"] == "pending_gpu_preflight"
    )
    with pytest.raises(PreflightError, match="below_one"):
        project(
            records, evidence_scope="target_single_vm_measured", two_vm_efficiency=1.0
        )


def test_worker_crash_isolated_and_target_requires_bounded_authorization(tmp_path):
    repo = root(tmp_path)
    crashed = plan()
    next(x for x in crashed["tasks"] if x["mode"] == "cpu_parallel_2")[
        "fixture_behavior"
    ] = "crash"
    crashed["plan_digest"] = plan_digest(crashed)
    output = repo / runner.ARTIFACT_ROOT / "fixture-crash"
    result = runner.run(
        crashed,
        profile(),
        output,
        mode="cpu_parallel_2",
        repo_root=repo,
        fixture=True,
        max_tasks=3,
        timeout_seconds=10,
    )
    assert (
        result["failed"] == 1
        and result["completed"] >= 1
        and not (output / "COMPLETED.json").exists()
    )
    target = profile("intended_single_vm_target")
    guard = runner.execution_guard(
        plan=plan(),
        profile=target,
        output_dir=repo / runner.ARTIFACT_ROOT / "target",
        fixture=False,
        bounded_authorized=False,
        repo_root=repo,
    )
    assert (
        not guard["authorized"]
        and "bounded_preflight_authorization_missing" in guard["reason_codes"]
    )
