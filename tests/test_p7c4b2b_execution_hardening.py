from __future__ import annotations

import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from creditrep.config.loader import sha256_canonical
from creditrep.experiments import p7c4b2b_preflight as runner
from creditrep.protocols.p7c4b2b import PreflightError


def _child_with_grandchild(pid_queue):
    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_queue.put(grandchild.pid)
    time.sleep(60)


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    tasks = [{"task_id": f"task-{index:02d}"} for index in range(54)]
    manifest = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "mode": "cpu_parallel_1",
        "plan_digest": "1" * 64,
        "machine_profile_digest": "2" * 64,
        "authorization_digest": "3" * 64,
        "proposal_digest": "4" * 64,
        "target_environment_digest": "5" * 64,
        "resource_policy_digest": "6" * 64,
        "physical_output_directory": str(run_dir.resolve()),
        "expected_tasks": tasks,
    }
    runner._atomic_json(run_dir / "run_manifest.json", manifest)
    runner._append_runtime_generation(run_dir, manifest, elapsed=0.0, failure_count=0)
    return run_dir, manifest


def _rewrite(path: Path, mutation) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    value["state_digest"] = runner._runtime_digest(value)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("accumulated_elapsed_seconds", -1.0),
        lambda value: value.__setitem__(
            "original_started_at_utc", "2020-01-01T00:00:00Z"
        ),
        lambda value: value.__setitem__("generation", 9),
        lambda value: value.__setitem__("previous_generation_digest", "0" * 64),
    ],
)
def test_runtime_ledger_redigested_mutations_fail_closed(tmp_path, mutation):
    run_dir, manifest = _manifest(tmp_path)
    runner._append_runtime_generation(run_dir, manifest, elapsed=1.0, failure_count=0)
    _rewrite(run_dir / "runtime-ledger" / "generation-000001.json", mutation)
    with pytest.raises(PreflightError, match="runtime_ledger"):
        runner._validate_runtime_ledger(run_dir, manifest)


def test_runtime_ledger_missing_middle_and_manifest_head_fail_closed(tmp_path):
    run_dir, manifest = _manifest(tmp_path)
    runner._append_runtime_generation(run_dir, manifest, elapsed=1.0, failure_count=0)
    runner._append_runtime_generation(run_dir, manifest, elapsed=2.0, failure_count=0)
    (run_dir / "runtime-ledger" / "generation-000001.json").unlink()
    with pytest.raises(PreflightError, match="runtime_ledger"):
        runner._validate_runtime_ledger(run_dir, manifest)


def test_runtime_generation_crash_windows_fail_closed(tmp_path, monkeypatch):
    run_dir, manifest = _manifest(tmp_path)
    original_atomic_json = runner._atomic_json

    def fail_manifest(path, value):
        if path.name == "run_manifest.json":
            raise OSError("injected manifest mirror failure")
        return original_atomic_json(path, value)

    monkeypatch.setattr(runner, "_atomic_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest"):
        runner._append_runtime_generation(
            run_dir, manifest, elapsed=1.0, failure_count=0
        )
    assert (run_dir / "runtime-ledger" / "generation-000001.json").is_file()
    persisted_manifest = runner._strict_json_file(run_dir / "run_manifest.json")
    with pytest.raises(PreflightError, match="runtime_ledger_integrity_failure"):
        runner._validate_runtime_ledger(run_dir, persisted_manifest)


def test_runtime_generation_failure_precedes_manifest_mirror(tmp_path, monkeypatch):
    run_dir, manifest = _manifest(tmp_path)
    checkpoint = dict(manifest["runtime_checkpoint"])
    original_create = runner._atomic_create_json

    def fail_generation(path, value):
        if path.parent.name == "runtime-ledger":
            raise OSError("injected generation failure")
        return original_create(path, value)

    monkeypatch.setattr(runner, "_atomic_create_json", fail_generation)
    with pytest.raises(OSError, match="injected generation"):
        runner._append_runtime_generation(
            run_dir, manifest, elapsed=1.0, failure_count=0
        )
    assert manifest["runtime_checkpoint"] == checkpoint
    assert not (run_dir / "runtime-ledger" / "generation-000001.json").exists()
    runner._validate_runtime_ledger(run_dir, manifest)


def test_uncommitted_runtime_temp_is_ignored_without_granting_allowance(tmp_path):
    run_dir, manifest = _manifest(tmp_path)
    temp = run_dir / "runtime-ledger" / ".generation-000001.json.crash.tmp"
    temp.write_text('{"accumulated_elapsed_seconds": 999999}', encoding="utf-8")
    head = runner._validate_runtime_ledger(run_dir, manifest)
    assert head["generation"] == 0
    appended = runner._append_runtime_generation(
        run_dir, manifest, elapsed=1.0, failure_count=0
    )
    assert appended["generation"] == 1
    assert appended["accumulated_elapsed_seconds"] == 1.0

    run_dir, manifest = _manifest(tmp_path / "other")
    manifest["runtime_checkpoint"]["state_digest"] = "f" * 64
    with pytest.raises(PreflightError, match="runtime_ledger"):
        runner._validate_runtime_ledger(run_dir, manifest)


def _write_failure(
    run_dir: Path, task_id: str, attempt: int, record: dict
) -> tuple[str, str]:
    quarantine = run_dir / "quarantine" / task_id / f"attempt-{attempt}"
    quarantine.mkdir(parents=True)
    runner._atomic_create_json(quarantine / "telemetry.json", record)
    evidence = {
        "task_id": task_id,
        "attempt": attempt,
        "reason_code": "transient_failure",
        "record_digest": sha256_canonical(record),
        "quarantine_path": str(quarantine.relative_to(run_dir)),
    }
    runner._atomic_create_json(
        run_dir / "attempt-failures" / task_id / f"attempt-{attempt}.json", evidence
    )
    return sha256_canonical(record), sha256_canonical(evidence)


def test_transient_failure_then_success_has_one_canonical_result(tmp_path):
    run_dir, manifest = _manifest(tmp_path)
    task_id = "task-00"
    started = runner._utc()
    runner._append_attempt_event(
        run_dir, manifest, task_id, 1, "running", started_at_utc=started
    )
    failed_record = {"task_id": task_id, "attempt": 1, "status": "failed"}
    telemetry_digest, failure_digest = _write_failure(
        run_dir, task_id, 1, failed_record
    )
    runner._append_attempt_event(
        run_dir,
        manifest,
        task_id,
        1,
        "transient_failure",
        started_at_utc=started,
        completed_at_utc=runner._utc(),
        reason_code="transient_failure",
        telemetry_digest=telemetry_digest,
        failure_evidence_digest=failure_digest,
    )
    runner._append_runtime_generation(run_dir, manifest, elapsed=1.0, failure_count=1)

    started = runner._utc()
    runner._append_attempt_event(
        run_dir, manifest, task_id, 2, "running", started_at_utc=started
    )
    result = {"task_id": task_id, "attempt": 2, "status": "completed"}
    attempt_dir = run_dir / "fits" / task_id / "attempts" / "attempt-2"
    attempt_dir.mkdir(parents=True)
    runner._atomic_create_json(attempt_dir / "telemetry.json", result)
    runner._atomic_create_json(run_dir / "fits" / task_id / "result.json", result)
    runner._append_attempt_event(
        run_dir,
        manifest,
        task_id,
        2,
        "succeeded",
        started_at_utc=started,
        completed_at_utc=runner._utc(),
        telemetry_digest=sha256_canonical(result),
        success_evidence_digest=sha256_canonical(result),
    )
    runner._append_runtime_generation(run_dir, manifest, elapsed=2.0, failure_count=1)

    histories, _heads = runner._validate_attempt_ledgers(
        run_dir, manifest, allow_running=False
    )
    assert {event["attempt_number"] for event in histories[task_id]} == {1, 2}
    assert sum(event["disposition"] == "succeeded" for event in histories[task_id]) == 1
    assert len(manifest["expected_tasks"]) == 54
    assert len(list((run_dir / "fits").glob("*/result.json"))) == 1


def test_deleted_failure_evidence_is_rejected(tmp_path):
    run_dir, manifest = _manifest(tmp_path)
    task_id = "task-00"
    started = runner._utc()
    runner._append_attempt_event(
        run_dir, manifest, task_id, 1, "running", started_at_utc=started
    )
    telemetry, evidence = _write_failure(
        run_dir, task_id, 1, {"task_id": task_id, "attempt": 1}
    )
    runner._append_attempt_event(
        run_dir,
        manifest,
        task_id,
        1,
        "transient_failure",
        started_at_utc=started,
        completed_at_utc=runner._utc(),
        telemetry_digest=telemetry,
        failure_evidence_digest=evidence,
    )
    (run_dir / "attempt-failures" / task_id / "attempt-1.json").unlink()
    with pytest.raises(PreflightError, match="control_file_mismatch"):
        runner._validate_attempt_ledgers(run_dir, manifest)


def test_rss_deduplicates_and_fails_closed(monkeypatch):
    class Process:
        pid = 7

        def children(self, recursive):
            assert recursive
            return [self, self]

        def memory_info(self):
            return type("Memory", (), {"rss": 11})()

    monkeypatch.setattr(runner.psutil, "Process", lambda _pid=None: Process())
    assert runner._aggregate_process_tree_rss(7) == 11

    def denied(self):
        raise psutil.AccessDenied(7)

    monkeypatch.setattr(Process, "memory_info", denied)
    with pytest.raises(PreflightError, match="memory_sampler_failure"):
        runner._aggregate_process_tree_rss(7)

    def operating_system_error(self):
        raise OSError("injected sampler failure")

    monkeypatch.setattr(Process, "memory_info", operating_system_error)
    with pytest.raises(PreflightError, match="memory_sampler_failure"):
        runner._aggregate_process_tree_rss(7)


def test_rss_disappearance_is_benign_only_when_pid_is_gone(monkeypatch):
    class Process:
        pid = 9

        def children(self, recursive):
            return []

        def memory_info(self):
            raise psutil.NoSuchProcess(9)

    monkeypatch.setattr(runner.psutil, "Process", lambda _pid=None: Process())
    monkeypatch.setattr(runner.psutil, "pid_exists", lambda _pid: False)
    assert runner._aggregate_process_tree_rss(9) == 0
    monkeypatch.setattr(runner.psutil, "pid_exists", lambda _pid: True)
    with pytest.raises(PreflightError, match="memory_sampler_failure"):
        runner._aggregate_process_tree_rss(9)


def test_real_spawned_child_and_grandchild_are_terminated_and_reaped():
    context = get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_child_with_grandchild, args=(queue,))
    process.start()
    grandchild_pid = queue.get(timeout=10)
    assert psutil.pid_exists(process.pid)
    assert psutil.pid_exists(grandchild_pid)
    assert runner._terminate_and_reap([process])
    assert not process.is_alive()
    for _ in range(20):
        if not psutil.pid_exists(grandchild_pid):
            break
        time.sleep(0.05)
    assert not psutil.pid_exists(grandchild_pid)


def test_control_file_mutation_and_partial_json_fail_closed(tmp_path):
    value = {"profile_digest": "a" * 64, "value": 1}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    binding = runner._control_binding(path, value, value["profile_digest"])
    runner._revalidate_control_files(
        {"machine_profile": binding}, {"machine_profile": value}
    )
    path.write_text('{"profile_digest":', encoding="utf-8")
    with pytest.raises(PreflightError, match="control_file_mismatch"):
        runner._revalidate_control_files(
            {"machine_profile": binding}, {"machine_profile": value}
        )


def test_output_traversal_and_different_physical_target_are_rejected(tmp_path):
    repo = tmp_path / "repo"
    authorized = repo / runner.ARTIFACT_ROOT
    authorized.mkdir(parents=True)
    valid = authorized / "run"
    identity = runner._resolved_output_identity(valid, repo)
    assert identity["physical_output_directory"] == str(valid.resolve())
    with pytest.raises(PreflightError, match="invalid_artifact_namespace"):
        runner._resolved_output_identity(authorized / ".." / "escape", repo)
    with pytest.raises(PreflightError, match="invalid_artifact_namespace"):
        runner._resolved_output_identity(tmp_path / "different" / "run", repo)


def test_output_symlink_collision_and_parent_retarget_fail_closed(tmp_path):
    repo = tmp_path / "repo"
    artifact_parent = repo / "artifacts"
    physical_one = tmp_path / "physical-one"
    physical_two = tmp_path / "physical-two"
    physical_one.mkdir()
    physical_two.mkdir()
    link = artifact_parent / "p7c4b2b-compute-preflight"
    artifact_parent.mkdir(parents=True)
    try:
        link.symlink_to(physical_one, target_is_directory=True)
    except OSError:
        pytest.skip("platform cannot create directory symlinks")
    output = link / "run"
    identity = runner._resolved_output_identity(output, repo)
    physical_output = Path(identity["physical_output_directory"])
    physical_output.mkdir()
    bound_output = runner._bind_physical_output(output, identity, repo)
    assert bound_output == physical_output
    link.unlink()
    link.symlink_to(physical_two, target_is_directory=True)
    (bound_output / "physical-write.json").write_text("{}", encoding="utf-8")
    assert (physical_one / "run" / "physical-write.json").is_file()
    assert not (physical_two / "run" / "physical-write.json").exists()
    with pytest.raises(PreflightError, match="output_identity_mismatch"):
        runner._revalidate_output_identity(bound_output, identity, repo)

    collision = physical_two / "collision-target"
    collision.mkdir()
    output_symlink = link / "collision"
    output_symlink.symlink_to(collision, target_is_directory=True)
    with pytest.raises(PreflightError, match="output_symlink_collision"):
        runner._resolved_output_identity(output_symlink, repo)


def test_interrupted_attempt_is_quarantined_and_accounted(tmp_path):
    run_dir, manifest = _manifest(tmp_path)
    task = manifest["expected_tasks"][0]
    started = runner._utc()
    runner._append_attempt_event(
        run_dir, manifest, task["task_id"], 1, "running", started_at_utc=started
    )
    state = runner._append_runtime_generation(
        run_dir, manifest, elapsed=0.5, failure_count=0
    )
    temporary = runner._target_attempt_temp(run_dir, task["task_id"], 1)
    telemetry = {"task_id": task["task_id"], "attempt": 1, "status": "running"}
    runner._atomic_create_json(temporary / "telemetry.json", telemetry)

    state = runner._reconcile_interrupted_attempts(
        run_dir, manifest, [task], state, elapsed=1.0
    )

    assert state["failure_count"] == 1
    assert not temporary.exists()
    assert not list(run_dir.rglob("*.tmp"))
    evidence_path = run_dir / "attempt-failures" / task["task_id"] / "attempt-1.json"
    assert evidence_path.is_file()
    histories, _ = runner._validate_attempt_ledgers(
        run_dir, manifest, allow_running=False
    )
    terminal = histories[task["task_id"]][-1]
    assert terminal["disposition"] == "interrupted"
    assert terminal["failure_evidence_digest"] == sha256_canonical(
        runner._strict_json_file(evidence_path)
    )


def test_interrupted_attempt_with_canonical_fit_fails_without_repair(tmp_path):
    run_dir, manifest = _manifest(tmp_path)
    task = manifest["expected_tasks"][0]
    runner._append_attempt_event(
        run_dir,
        manifest,
        task["task_id"],
        1,
        "running",
        started_at_utc=runner._utc(),
    )
    state = runner._append_runtime_generation(
        run_dir, manifest, elapsed=0.5, failure_count=0
    )
    canonical = run_dir / "fits" / task["task_id"]
    canonical.mkdir(parents=True)
    before = sorted(
        (path.relative_to(run_dir), path.read_bytes())
        for path in run_dir.rglob("*.json")
    )
    with pytest.raises(
        PreflightError, match="canonical_promotion_without_attempt_commit"
    ):
        runner._reconcile_interrupted_attempts(
            run_dir, manifest, [task], state, elapsed=1.0
        )
    after = sorted(
        (path.relative_to(run_dir), path.read_bytes())
        for path in run_dir.rglob("*.json")
    )
    assert after == before


def _promotion_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    run_dir = repo / runner.ARTIFACT_ROOT / "run"
    run_dir.mkdir(parents=True)
    task = {"task_id": "task-00"}
    controls = {
        "machine_profile": {"profile_digest": "1" * 64},
        "target_environment": {"environment_digest": "2" * 64},
        "authorization_proposal": {"proposal_digest": "3" * 64},
        "effective_authorization": {"authorization_digest": "4" * 64},
    }
    control_files = {}
    for name, value in controls.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        digest = next(item for key, item in value.items() if key.endswith("digest"))
        control_files[name] = runner._control_binding(path, value, digest)
    manifest = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "mode": "cpu_parallel_1",
        "plan_digest": "5" * 64,
        "machine_profile_digest": "1" * 64,
        "authorization_digest": "4" * 64,
        "proposal_digest": "3" * 64,
        "target_environment_digest": "2" * 64,
        "resource_policy_digest": "6" * 64,
        "physical_output_directory": str(run_dir.resolve()),
        "output_identity": runner._resolved_output_identity(run_dir, repo),
        "control_files": control_files,
        "expected_tasks": [task],
    }
    runner._atomic_json(run_dir / "run_manifest.json", manifest)
    runner._append_runtime_generation(run_dir, manifest, elapsed=0.0, failure_count=0)
    started = runner._utc()
    runner._append_attempt_event(
        run_dir, manifest, task["task_id"], 1, "running", started_at_utc=started
    )
    runner._append_runtime_generation(run_dir, manifest, elapsed=0.1, failure_count=0)
    temporary = runner._target_attempt_temp(run_dir, task["task_id"], 1)
    runner._atomic_create_json(
        temporary / "telemetry.json", {"task_id": task["task_id"]}
    )
    plan = {
        "limits": {
            "global_wall_clock_seconds": {"value": 43_200},
            "aggregate_process_tree_rss_bytes": {"value": 12_348_030_976},
            "minimum_system_available_ram_bytes": {"value": 2 * 1024**3},
            "minimum_free_disk_bytes": {"value": 16 * 1024**3},
            "artifact_size_bytes": {"value": 2 * 1024**3},
        }
    }
    return repo, run_dir, temporary, task, manifest, controls, plan


@pytest.mark.parametrize(
    "case,code",
    [
        ("expiry", "authorization_expired"),
        ("runtime", "global_wall_clock_exceeded"),
        ("timeout", "fit_timeout"),
        ("rss", "aggregate_memory_guard_triggered"),
        ("ram", "insufficient_available_ram"),
        ("disk", "insufficient_free_disk"),
        ("size", "artifact_size_guard_triggered"),
        ("control", "control_file_mismatch"),
        ("path", "output_identity_mismatch"),
    ],
)
def test_each_late_promotion_guard_blocks_canonical_marker(
    tmp_path, monkeypatch, case, code
):
    repo, run_dir, temporary, task, manifest, controls, plan = _promotion_fixture(
        tmp_path
    )
    report = {"valid": True, "reason_codes": []}
    if case == "expiry":
        report = {"valid": False, "reason_codes": ["authorization_expired"]}
    monkeypatch.setattr(
        runner, "validate_effective_authorization", lambda *_a, **_k: report
    )
    available = 1 if case == "ram" else 64 * 1024**3
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: type("VM", (), {"available": available})(),
    )
    if case == "size":
        plan["limits"]["artifact_size_bytes"]["value"] = 1
    if case == "control":
        path = Path(manifest["control_files"]["machine_profile"]["source_path"])
        path.write_text("{}", encoding="utf-8")
    if case == "path":
        manifest["output_identity"]["physical_parent"] = str(tmp_path / "wrong")
    elapsed = 43_200 if case == "runtime" else 1.0
    timed_out = case == "timeout"
    rss = (
        plan["limits"]["aggregate_process_tree_rss_bytes"]["value"] + 1
        if case == "rss"
        else 1
    )
    free = 1 if case == "disk" else 64 * 1024**3
    with pytest.raises(PreflightError, match=code):
        runner._prepromotion_guards(
            run_dir=run_dir,
            temporary=temporary,
            task=task,
            attempt=1,
            manifest=manifest,
            plan=plan,
            profile=controls["machine_profile"],
            target_environment=controls["target_environment"],
            authorization_proposal=controls["authorization_proposal"],
            effective_authorization=controls["effective_authorization"],
            control_values=controls,
            repo_root=repo,
            elapsed=elapsed,
            timed_out=timed_out,
            rss_sampler=lambda _pid: rss,
            disk_usage_provider=lambda _path: type("Disk", (), {"free": free})(),
        )
    assert not (run_dir / "fits" / task["task_id"]).exists()
    assert not (run_dir / "fits" / task["task_id"] / "COMPLETED.json").exists()
