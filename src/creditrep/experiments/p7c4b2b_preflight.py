"""Executable bounded P7C.4B.2b harness. Development use is fixture-only."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import psutil

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.registry import find_repo_root
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2b import (
    DATASET_FINGERPRINTS,
    MODES,
    SCIENTIFIC_DIGEST,
    PreflightError,
    build_plan,
    machine_profile_digest,
    project,
    ram_feasibility,
    summarize,
    validate_machine,
    validate_plan,
)

ARTIFACT_ROOT = "artifacts/p7c4b2b-compute-preflight"
EVIDENCE_FIXTURE = "development_fixture_non_benchmark"
EXIT_OK, EXIT_VALIDATION, EXIT_GUARD, EXIT_CONFIG, EXIT_MISSING = 0, 2, 3, 4, 5
VALIDATION_CODES = {
    "manifest_digest_mismatch",
    "plan_digest_mismatch",
    "machine_profile_digest_mismatch",
    "foreign_machine_or_run",
    "missing_logical_unit",
    "duplicate_logical_unit",
    "duplicate_successful_attempt",
    "invalid_timing",
    "invalid_memory",
    "warmup_mixed_into_measured",
    "worker_limit_violation",
    "thread_limit_violation",
    "corrupt_json",
    "partial_temporary_attempt",
    "retry_identity_mismatch",
    "premature_completion_marker",
    "summary_mismatch",
    "projection_uses_b1_fixture",
    "projection_without_target_measurement",
    "cost_without_operator_price",
    "missing_provenance",
    "failed_unit_present",
    "artifact_size_violation",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read(path: Path, codes: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        codes.append("corrupt_json")
        return None


def capture_machine_profile(
    *,
    role: str = "development_calibration_only",
    provider: str = "operator_unspecified",
    instance_type: str = "operator_unspecified",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    disk = shutil.disk_usage(root / "artifacts")
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-C",
        str(root),
        "rev-parse",
        "HEAD",
    ]
    head = subprocess.check_output(command, text=True, encoding="utf-8").strip()
    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--local"], text=True, encoding="utf-8"
    )
    try:
        python_executable = Path(sys.executable).resolve().relative_to(root).as_posix()
    except ValueError:
        python_executable = Path(sys.executable).name
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    profile = {
        "schema_version": 1,
        "machine_role": role,
        "machine_id": sha256_canonical(
            {"node": platform.node(), "platform": platform.platform()}
        ),
        "cloud_provider": provider,
        "instance_type": instance_type,
        "os": platform.platform(),
        "cpu_model": platform.processor() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": vm.total,
        "system_used_ram_bytes": vm.used,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "disk_free_bytes": disk.free,
        "python_executable": python_executable,
        "python_version": platform.python_version(),
        "dependency_fingerprint": sha256_canonical(sorted(packages.splitlines())),
        "git_commit": head,
        "scientific_manifest_digest": SCIENTIFIC_DIGEST,
        "dataset_fingerprints": DATASET_FINGERPRINTS,
        "worker_limit": 2,
        "threads_per_worker": 2,
        "virtualization_power": "unknown",
        "utc_captured": _utc(),
    }
    profile["profile_digest"] = machine_profile_digest(profile)
    return profile


def actual_fit_adapter(task: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Actual contract boundary; imported lazily and never invoked by fixture tests."""
    import csv
    from creditrep.datasets import load_dataset
    from creditrep.experiments.model_validation import _fit_for_partition
    from creditrep.experiments.p7c3_feasibility import adapt_mlp_feasibility_candidate
    from creditrep.preprocessing import load_protocol_a_config
    from creditrep.splitting import create_nested_cv_definition

    dataset = load_dataset(task["dataset_id"], repo_root=repo_root)
    with (repo_root / "data" / "checksums-sha256.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        checksums = {row["Path"]: row["Hash"] for row in csv.DictReader(handle)}
    nested = create_nested_cv_definition(
        dataset,
        dataset_checksum=checksums[dataset.metadata["source_file"]],
        outer_n_repeats=1,
        outer_n_splits=2,
        inner_n_splits=5,
        random_seed=42,
    )
    outer = nested.outer_folds[0]
    inner = outer.inner_folds[0]
    candidate = {"id": task["candidate_id"], **task["candidate"]}
    parameters = adapt_mlp_feasibility_candidate(
        task["model_id"],
        candidate,
        {
            "optimizer": "adam",
            "batch_size": 32,
            "max_epochs": 200,
            "device_policy": "cpu",
            "early_stopping": {"enabled": True, "patience": 20, "min_delta": 0.0001},
        },
    )
    started = time.perf_counter()
    estimator, _, evaluation, _ = _fit_for_partition(
        dataset=dataset,
        model_id=task["model_id"],
        parameters=parameters,
        seed=task["seed_identity"],
        outer_id=outer.outer_fold_id,
        inner_id=inner.inner_fold_id,
        candidate_id=task["candidate_id"],
        train_indices=inner.train_indices,
        evaluation_indices=inner.validation_indices,
        protocol_config=load_protocol_a_config(repo_root=repo_root),
        model_stage="p7c4b2b_compute_preflight",
    )
    scoring_started = time.perf_counter()
    estimator.predict_proba(evaluation)
    return {
        "fit_pipeline_seconds": scoring_started - started,
        "validation_scoring_seconds": time.perf_counter() - scoring_started,
    }


def _thread_env() -> dict[str, str]:
    values = {
        key: "2"
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    values.update({"TOKENIZERS_PARALLELISM": "false"})
    os.environ.update(values)
    return values


def _worker(queue, task: dict[str, Any], root_text: str, fixture: bool) -> None:
    env = _thread_env()
    started_cpu = time.process_time()
    rss = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    try:
        if fixture:
            behavior = task.get("fixture_behavior", "success")
            if behavior == "sleep":
                time.sleep(float(task.get("fixture_sleep_seconds", 1)))
            elif behavior == "crash":
                os._exit(17)
            elif behavior == "transient":
                raise OSError("fixture transient")
            elif behavior == "failure":
                raise ValueError("fixture deterministic")
            detail = {"validation_scoring_seconds": 0.0, "fixture": True}
        else:
            detail = actual_fit_adapter(task, Path(root_text))
        queue.put(
            {
                "ok": True,
                "cpu_seconds": time.process_time() - started_cpu,
                "rss_start": rss,
                "rss_peak": psutil.Process().memory_info().rss,
                "available_start": available,
                "available_end": psutil.virtual_memory().available,
                "thread_env": env,
                "detail": detail,
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:300],
                "cpu_seconds": time.process_time() - started_cpu,
                "rss_start": rss,
                "rss_peak": psutil.Process().memory_info().rss,
                "available_start": available,
                "available_end": psutil.virtual_memory().available,
                "thread_env": env,
            }
        )


def execution_guard(
    *,
    plan: dict[str, Any],
    profile: dict[str, Any],
    output_dir: Path,
    fixture: bool,
    bounded_authorized: bool,
    repo_root: Path,
) -> dict[str, Any]:
    codes = []
    try:
        validate_plan(plan)
    except PreflightError:
        codes.append("invalid_preflight_plan")
    if fixture:
        if profile.get("machine_role") != "development_calibration_only":
            codes.append("fixture_machine_role_mismatch")
    else:
        try:
            validate_machine(profile, plan)
        except PreflightError as exc:
            codes.extend(str(exc).split(","))
        if not bounded_authorized:
            codes.append("bounded_preflight_authorization_missing")
    expected = (repo_root / ARTIFACT_ROOT).resolve()
    if output_dir.resolve().parent != expected or not output_dir.name:
        codes.append("invalid_artifact_namespace")
    if not fixture:
        disk_parent = expected if expected.exists() else expected.parent
        if (
            shutil.disk_usage(disk_parent).free
            < plan["limits"]["minimum_free_disk_bytes"]["value"]
        ):
            codes.append("insufficient_free_disk")
        vm = psutil.virtual_memory()
        if vm.available < plan["limits"]["minimum_system_available_ram_bytes"]["value"]:
            codes.append("insufficient_available_ram")
    return {"authorized": not codes, "reason_codes": sorted(set(codes))}


def _tasks(
    plan: dict[str, Any], mode: str, max_tasks: int | None
) -> list[dict[str, Any]]:
    selected = [deepcopy(x) for x in plan["tasks"] if x["mode"] == mode]
    return selected if max_tasks is None else selected[:max_tasks]


def _terminate_tree(process) -> bool:
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.terminate()
        _, alive = psutil.wait_procs(parent.children(recursive=True), timeout=1)
        for child in alive:
            child.kill()
    except psutil.Error:
        pass
    if process.is_alive():
        process.terminate()
        process.join(1)
    if process.is_alive():
        process.kill()
        process.join(1)
    return not process.is_alive()


def _aggregate_process_tree_rss() -> int:
    """Sample the runner tree while tolerating children that exit mid-sample."""
    parent = psutil.Process()
    total = 0
    for process in [parent, *parent.children(recursive=True)]:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total


def _promote_attempt(run: Path, task: dict[str, Any], record: dict[str, Any]) -> None:
    fit = run / "fits" / task["task_id"]
    attempt = int(record["attempt"])
    temporary = run / "temporary" / task["task_id"] / f"attempt-{attempt}.tmp"
    final = fit / "attempts" / f"attempt-{attempt}"
    temporary.mkdir(parents=True, exist_ok=False)
    _atomic_json(temporary / "telemetry.json", record)
    if record["task_id"] != task["task_id"] or record["attempt"] != attempt:
        raise PreflightError("retry_identity_mismatch")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    if record["status"] == "completed":
        if (fit / "COMPLETED.json").exists():
            raise PreflightError("duplicate_successful_attempt")
        _atomic_json(fit / "result.json", record)
        _atomic_json(
            fit / "COMPLETED.json",
            {
                "task_id": task["task_id"],
                "attempt": attempt,
                "record_digest": sha256_canonical(record),
            },
        )
    else:
        _atomic_json(fit / "failure.json", record)


def run(
    plan: dict[str, Any],
    profile: dict[str, Any],
    output_dir: Path,
    *,
    mode: str,
    repo_root: Path | None = None,
    fixture: bool = False,
    bounded_authorized: bool = False,
    max_tasks: int | None = None,
    timeout_seconds: float | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    output_dir = output_dir.resolve()
    guard = execution_guard(
        plan=plan,
        profile=profile,
        output_dir=output_dir,
        fixture=fixture,
        bounded_authorized=bounded_authorized,
        repo_root=root,
    )
    if not guard["authorized"]:
        raise PreflightError(",".join(guard["reason_codes"]))
    if not fixture and max_tasks is not None:
        raise PreflightError("fixture_bounds_forbidden_on_target")
    if output_dir.exists():
        return resume(
            plan,
            profile,
            output_dir,
            mode=mode,
            repo_root=root,
            fixture=fixture,
            bounded_authorized=bounded_authorized,
            max_tasks=max_tasks,
            timeout_seconds=timeout_seconds,
            fail_fast=fail_fast,
        )
    output_dir.mkdir(parents=True)
    expected = _tasks(plan, mode, max_tasks)
    _atomic_json(output_dir / "plan.json", plan)
    _atomic_json(output_dir / "machine_profile.json", profile)
    _atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": output_dir.name,
            "mode": mode,
            "evidence_scope": EVIDENCE_FIXTURE
            if fixture
            else "target_single_vm_measured",
            "scientific_manifest_digest": SCIENTIFIC_DIGEST,
            "plan_digest": plan["plan_digest"],
            "machine_profile_digest": profile["profile_digest"],
            "max_workers": MODES[mode],
            "threads_per_worker": 2,
            "expected_tasks": expected,
            "created_utc": _utc(),
        },
    )
    return resume(
        plan,
        profile,
        output_dir,
        mode=mode,
        repo_root=root,
        fixture=fixture,
        bounded_authorized=bounded_authorized,
        max_tasks=max_tasks,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
    )


def resume(
    plan: dict[str, Any],
    profile: dict[str, Any],
    run_dir: Path,
    *,
    mode: str,
    repo_root: Path | None = None,
    fixture: bool = False,
    bounded_authorized: bool = False,
    max_tasks: int | None = None,
    timeout_seconds: float | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    run_dir = run_dir.resolve()
    guard = execution_guard(
        plan=plan,
        profile=profile,
        output_dir=run_dir,
        fixture=fixture,
        bounded_authorized=bounded_authorized,
        repo_root=root,
    )
    if not guard["authorized"]:
        raise PreflightError(",".join(guard["reason_codes"]))
    manifest = _read(run_dir / "run_manifest.json", [])
    if (
        not manifest
        or manifest.get("plan_digest") != plan["plan_digest"]
        or manifest.get("machine_profile_digest") != profile["profile_digest"]
    ):
        raise PreflightError("incompatible_resume")
    expected = _tasks(plan, mode, max_tasks)
    pending = []
    skipped = 0
    for task in expected:
        fit = run_dir / "fits" / task["task_id"]
        if (fit / "result.json").is_file() and (fit / "COMPLETED.json").is_file():
            codes = []
            record = _read(fit / "result.json", codes)
            marker = _read(fit / "COMPLETED.json", codes)
            if (
                codes
                or not record
                or not marker
                or marker.get("record_digest") != sha256_canonical(record)
            ):
                raise PreflightError("corrupt_completed_artifact")
            skipped += 1
        elif fit.exists() and any(fit.rglob("*")):
            raise PreflightError("corrupt_completed_artifact")
        else:
            pending.append(task)
    timeout = float(
        timeout_seconds or plan["limits"]["per_fit_timeout_seconds"]["value"]
    )
    workers = MODES[mode]
    retry_max = plan["limits"]["retry_maximum"]["value"]
    context = get_context("spawn")
    active = {}
    attempts = {}
    completed = failed = 0
    started_global = time.monotonic()
    peak_aggregate = 0
    while pending or active:
        if (
            time.monotonic() - started_global
            > plan["limits"]["global_wall_clock_seconds"]["value"]
        ):
            raise PreflightError("global_wall_clock_exceeded")
        while pending and len(active) < workers:
            task = pending.pop(0)
            attempts[task["task_id"]] = attempts.get(task["task_id"], 0) + 1
            q = context.Queue()
            p = context.Process(target=_worker, args=(q, task, str(root), fixture))
            p.start()
            active[task["task_id"]] = (p, q, task, time.monotonic(), _utc())
        peak_aggregate = max(peak_aggregate, _aggregate_process_tree_rss())
        if (
            peak_aggregate > plan["limits"]["aggregate_process_tree_rss_bytes"]["value"]
            or psutil.virtual_memory().available
            < plan["limits"]["minimum_system_available_ram_bytes"]["value"]
        ):
            for p, *_ in active.values():
                _terminate_tree(p)
            raise PreflightError("aggregate_memory_guard_triggered")
        finished = []
        for task_id, (p, q, task, started, started_utc) in active.items():
            if not p.is_alive() or time.monotonic() - started >= timeout:
                finished.append(task_id)
        if not finished:
            time.sleep(0.01)
            continue
        for task_id in finished:
            p, q, task, started, started_utc = active.pop(task_id)
            timed_out = p.is_alive()
            cleaned = (
                _terminate_tree(p)
                if timed_out
                else (p.join(1) is None and not p.is_alive())
            )
            msg = (
                q.get()
                if not q.empty()
                else {
                    "ok": False,
                    "error_type": "WorkerCrash",
                    "message": "worker exited without telemetry",
                    "cpu_seconds": 0,
                    "rss_start": 0,
                    "rss_peak": 0,
                    "available_start": psutil.virtual_memory().available,
                    "available_end": psutil.virtual_memory().available,
                    "thread_env": {},
                }
            )
            transient = msg.get("error_type") in {"OSError", "ConnectionError"}
            reason = (
                "fit_timeout"
                if timed_out
                else None
                if msg.get("ok") and p.exitcode == 0
                else "transient_failure"
                if transient
                else "worker_crash"
                if msg.get("error_type") == "WorkerCrash"
                else "deterministic_failure"
            )
            record = {
                **task,
                "schema_version": 1,
                "run_id": run_dir.name,
                "scenario_id": mode,
                "attempt": attempts[task_id],
                "worker_identity": p.pid,
                "started_utc": started_utc,
                "completed_utc": _utc(),
                "started_monotonic": started,
                "completed_monotonic": time.monotonic(),
                "wall_clock_seconds": time.monotonic() - started,
                "cpu_time_seconds": msg.get("cpu_seconds", 0),
                "status": "completed" if reason is None else "failed",
                "reason_code": reason,
                "process_exit_code": p.exitcode,
                "process_rss_start_bytes": msg.get("rss_start", 0),
                "process_peak_rss_bytes": msg.get("rss_peak", 0),
                "process_rss_delta_bytes": max(
                    0, msg.get("rss_peak", 0) - msg.get("rss_start", 0)
                ),
                "aggregate_process_tree_peak_rss_bytes": peak_aggregate,
                "system_available_ram_start_bytes": msg.get("available_start", 0),
                "system_available_ram_min_bytes": min(
                    msg.get("available_start", 0), msg.get("available_end", 0)
                ),
                "system_available_ram_end_bytes": msg.get("available_end", 0),
                "max_workers": workers,
                "threads_per_worker": 2,
                "thread_env": msg.get("thread_env", {}),
                "machine_profile_digest": profile["profile_digest"],
                "preflight_plan_digest": plan["plan_digest"],
                "orphan_cleanup_pass": cleaned,
            }
            _promote_attempt(run_dir, task, record)
            artifact_bytes = sum(
                path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
            )
            if artifact_bytes > plan["limits"]["artifact_size_bytes"]["value"]:
                raise PreflightError("artifact_size_guard_triggered")
            if reason is None:
                completed += 1
            elif transient and attempts[task_id] <= retry_max:
                pending.insert(0, task)
            else:
                failed += 1
                if fail_fast:
                    for child, *_ in active.values():
                        _terminate_tree(child)
                    active.clear()
                    pending.clear()
                    break
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (run_dir / "fits").glob("*/result.json")
    ]
    summary = (
        summarize(records, mode=mode)
        if any(x.get("classification") == "measured" for x in records)
        else {
            "mode": mode,
            "measured_count": 0,
            "warmups_excluded": sum(
                x.get("classification") == "warmup" for x in records
            ),
        }
    )
    _atomic_json(run_dir / "summary.json", summary)
    projection = project(records, evidence_scope=manifest["evidence_scope"])
    _atomic_json(run_dir / "projection.json", projection)
    _atomic_json(
        run_dir / "ram_feasibility.json",
        ram_feasibility(records, profile, evidence_scope=manifest["evidence_scope"]),
    )
    report = validate_artifacts(
        run_dir, allow_incomplete=failed > 0 or len(records) < len(expected)
    )
    _atomic_json(run_dir / "validation_report.json", report)
    if report["valid"] and len(records) == len(expected):
        _atomic_json(
            run_dir / "COMPLETED.json",
            {
                "run_id": run_dir.name,
                "validation_report_digest": sha256_canonical(report),
            },
        )
    return {
        "run_id": run_dir.name,
        "executed": completed + failed,
        "skipped": skipped,
        "completed": completed,
        "failed": failed,
        "validation": report,
    }


def validate_artifacts(
    run_dir: Path, *, allow_incomplete: bool = False
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    codes = []
    if not run_dir.is_dir():
        return {"valid": False, "reason_codes": ["missing_provenance"]}
    artifact_bytes = sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    if any(p.name.endswith(".tmp") for p in run_dir.rglob("*")):
        codes.append("partial_temporary_attempt")
    manifest = _read(run_dir / "run_manifest.json", codes)
    plan = _read(run_dir / "plan.json", codes)
    profile = _read(run_dir / "machine_profile.json", codes)
    if manifest is None or plan is None or profile is None:
        return {
            "valid": False,
            "reason_codes": sorted(set(codes + ["missing_provenance"])),
        }
    if artifact_bytes > plan.get("limits", {}).get("artifact_size_bytes", {}).get(
        "value", 0
    ):
        codes.append("artifact_size_violation")
    if manifest.get("scientific_manifest_digest") != SCIENTIFIC_DIGEST:
        codes.append("manifest_digest_mismatch")
    if manifest.get("plan_digest") != plan.get("plan_digest") or plan.get(
        "plan_digest"
    ) != sha256_canonical({k: v for k, v in plan.items() if k != "plan_digest"}):
        codes.append("plan_digest_mismatch")
    if manifest.get("machine_profile_digest") != profile.get(
        "profile_digest"
    ) or profile.get("profile_digest") != machine_profile_digest(profile):
        codes.append("machine_profile_digest_mismatch")
    if manifest.get("run_id") != run_dir.name:
        codes.append("foreign_machine_or_run")
    if manifest.get("max_workers", 3) > 2:
        codes.append("worker_limit_violation")
    expected = manifest.get("expected_tasks", [])
    expected_ids = [x.get("task_id") for x in expected]
    expected_by_id = {x.get("task_id"): x for x in expected}
    if len(expected_ids) != len(set(expected_ids)):
        codes.append("duplicate_logical_unit")
    actual_dirs = (
        [x for x in (run_dir / "fits").glob("*") if x.is_dir()]
        if (run_dir / "fits").is_dir()
        else []
    )
    actual_ids = [x.name for x in actual_dirs]
    if set(actual_ids) - set(expected_ids):
        codes.append("foreign_machine_or_run")
    if not allow_incomplete and set(expected_ids) - set(actual_ids):
        codes.append("missing_logical_unit")
    records = []
    for directory in actual_dirs:
        if (directory / "failure.json").exists():
            codes.append("failed_unit_present")
        result = _read(directory / "result.json", codes)
        if not result:
            continue
        records.append(result)
        if (
            result.get("task_id") != directory.name
            or result.get("run_id") != run_dir.name
        ):
            codes.append("foreign_machine_or_run")
        expected_task = expected_by_id.get(result.get("task_id"), {})
        for field in (
            "logical_unit_id",
            "dataset_id",
            "dataset_fingerprint",
            "model_id",
            "candidate_id",
            "seed_identity",
            "classification",
            "mode",
            "repetition",
        ):
            if result.get(field) != expected_task.get(field):
                codes.append("retry_identity_mismatch")
                break
        if result.get("preflight_plan_digest") != plan.get("plan_digest") or result.get(
            "machine_profile_digest"
        ) != profile.get("profile_digest"):
            codes.append("retry_identity_mismatch")
        if (
            not isinstance(result.get("wall_clock_seconds"), (int, float))
            or result["wall_clock_seconds"] < 0
            or result.get("completed_monotonic", 0) < result.get("started_monotonic", 0)
        ):
            codes.append("invalid_timing")
        memory = [
            result.get(key)
            for key in (
                "process_rss_start_bytes",
                "process_peak_rss_bytes",
                "aggregate_process_tree_peak_rss_bytes",
                "system_available_ram_min_bytes",
            )
        ]
        if (
            any(not isinstance(x, (int, float)) or x < 0 for x in memory)
            or (
                isinstance(result.get("process_peak_rss_bytes"), (int, float))
                and result.get("process_peak_rss_bytes", 0)
                < result.get("process_rss_start_bytes", 0)
            )
            or result.get("system_available_ram_min_bytes", 0)
            > profile.get("ram_total_bytes", 0)
        ):
            codes.append("invalid_memory")
        if result.get("max_workers", 3) > 2:
            codes.append("worker_limit_violation")
        if result.get("orphan_cleanup_pass") is not True:
            codes.append("foreign_machine_or_run")
        if result.get("threads_per_worker") != 2 or any(
            result.get("thread_env", {}).get(key) != "2"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            codes.append("thread_limit_violation")
        attempts = (
            list((directory / "attempts").glob("attempt-*/telemetry.json"))
            if (directory / "attempts").is_dir()
            else []
        )
        successes = 0
        for path in attempts:
            attempt = _read(path, codes)
            if attempt and attempt.get("task_id") != result.get("task_id"):
                codes.append("retry_identity_mismatch")
            successes += int(bool(attempt and attempt.get("status") == "completed"))
        if successes > 1:
            codes.append("duplicate_successful_attempt")
        marker = _read(directory / "COMPLETED.json", codes)
        if marker and marker.get("record_digest") != sha256_canonical(result):
            codes.append("premature_completion_marker")
    stored = _read(run_dir / "summary.json", codes)
    if stored and any(x.get("classification") == "measured" for x in records):
        try:
            rebuilt = summarize(records, mode=manifest["mode"])
        except PreflightError:
            rebuilt = None
        if rebuilt is None or stored != rebuilt:
            codes.append("summary_mismatch")
        if stored.get("warmups_excluded", 0) and stored.get(
            "measured_count", 0
        ) + stored["warmups_excluded"] != len(records):
            codes.append("warmup_mixed_into_measured")
    projection = _read(run_dir / "projection.json", codes)
    if projection:
        if (
            manifest.get("evidence_scope") == EVIDENCE_FIXTURE
            and projection.get("status") != "pending_target_measurement"
        ):
            codes.append("projection_uses_b1_fixture")
        if manifest.get(
            "evidence_scope"
        ) != "target_single_vm_measured" and projection.get("status") not in {
            "pending_target_measurement",
            None,
        }:
            codes.append("projection_without_target_measurement")
        if projection.get("cost", {}).get("status") not in {
            "pending_operator_price_input",
            None,
        }:
            codes.append("cost_without_operator_price")
    completion = (
        _read(run_dir / "COMPLETED.json", codes)
        if (run_dir / "COMPLETED.json").exists()
        else None
    )
    if completion:
        stored_report = _read(run_dir / "validation_report.json", codes)
        if (
            set(expected_ids) != set(actual_ids)
            or codes
            or not stored_report
            or completion.get("validation_report_digest")
            != sha256_canonical(stored_report)
        ):
            codes.append("premature_completion_marker")
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "valid": not codes,
        "reason_codes": sorted(set(codes)),
        "expected": len(expected_ids),
        "completed": len(records),
    }


def load_default_plan(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    manifest = load_manifest(
        root / "configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml",
        repo_root=root,
    )
    return build_plan(manifest)
