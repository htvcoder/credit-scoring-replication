"""Executable bounded P7C.4B.2b harness. Development use is fixture-only."""

from __future__ import annotations

import json
from queue import Empty
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import psutil

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.registry import find_repo_root
from creditrep.process_tree import (
    close_process_queue,
    isolate_process_group,
    terminate_and_reap,
)
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
    summarize_legacy_v1,
    validate_machine,
    validate_plan,
)
from creditrep.protocols.p7c4b2b_authorization import (
    EXECUTION_STAGE as TARGET_INNER_EXECUTION_STAGE,
    validate_authorization_proposal,
    validate_effective_authorization,
    validate_target_environment,
)

ARTIFACT_ROOT = "artifacts/p7c4b2b-compute-preflight"
EVIDENCE_FIXTURE = "development_fixture_non_benchmark"
RUNTIME_LEDGER_ARTIFACT_TYPE = "p7c4b2b_runtime_ledger_generation"
RUNTIME_LEDGER_SCHEMA_VERSION = 2
ATTEMPT_LEDGER_ARTIFACT_TYPE = "p7c4b2b_task_attempt_event"
ATTEMPT_LEDGER_SCHEMA_VERSION = 1
RUNTIME_CLOCK_SKEW_SECONDS = 5.0
PROMOTION_OVERHEAD_BYTES = 4096
TRANSIENT_REASON_CODES = frozenset({"transient_failure"})
ATTEMPT_DISPOSITIONS = frozenset(
    {"running", "interrupted", "transient_failure", "permanent_failure", "succeeded"}
)
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
    "runtime_ledger_invalid",
    "runtime_ledger_integrity_failure",
    "runtime_clock_rollback",
    "attempt_ledger_invalid",
    "attempt_ledger_integrity_failure",
    "control_file_mismatch",
    "output_identity_mismatch",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise PreflightError("runtime_state_invalid") from exc
    if parsed.tzinfo is None:
        raise PreflightError("runtime_state_invalid")
    return parsed


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    """Durably create one JSON artifact without ever replacing an old one."""
    if path.exists() or path.is_symlink():
        raise PreflightError("artifact_collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PreflightError("artifact_collision") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PreflightError("control_file_mismatch") from exc


def _strict_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("control_file_mismatch") from exc
    if not isinstance(value, dict):
        raise PreflightError("control_file_mismatch")
    return value


def _control_binding(
    path: Path, expected: dict[str, Any], artifact_digest: str
) -> dict[str, Any]:
    canonical = path.absolute()
    source = path.resolve(strict=True)
    value = _strict_json_file(source)
    if value != expected:
        raise PreflightError("control_file_mismatch")
    return {
        "canonical_path": str(canonical),
        "source_path": str(source),
        "source_sha256": _file_sha256(source),
        "artifact_digest": artifact_digest,
    }


def _revalidate_control_files(
    bindings: dict[str, Any], expected: dict[str, dict[str, Any]]
) -> None:
    if set(bindings) != set(expected):
        raise PreflightError("control_file_mismatch")
    for name, value in expected.items():
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise PreflightError("control_file_mismatch")
        path = Path(str(binding.get("source_path", "")))
        canonical_path = Path(str(binding.get("canonical_path", "")))
        try:
            canonical_resolved = canonical_path.resolve(strict=True)
        except OSError as exc:
            raise PreflightError("control_file_mismatch") from exc
        current = _strict_json_file(path)
        digest_fields = {
            "machine_profile": "profile_digest",
            "target_environment": "environment_digest",
            "authorization_proposal": "proposal_digest",
            "effective_authorization": "authorization_digest",
        }
        digest_field = digest_fields.get(name)
        if (
            current != value
            or canonical_resolved != path
            or _file_sha256(path) != binding.get("source_sha256")
            or digest_field is None
            or current.get(digest_field) != binding.get("artifact_digest")
        ):
            raise PreflightError("control_file_mismatch")


def _read(path: Path, codes: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        codes.append("corrupt_json")
        return None


def _equivalent_json(left: Any, right: Any) -> bool:
    """Compare reconstructed JSON while tolerating harmless float sum ordering."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equivalent_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent_json(a, b) for a, b in zip(left, right)
        )
    return left == right


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolved_output_identity(output: Path, repo_root: Path) -> dict[str, str]:
    authorized_root = (repo_root / ARTIFACT_ROOT).resolve(strict=False)
    physical = output.resolve(strict=False)
    if (
        not output.name
        or physical == authorized_root
        or not _is_within(physical, authorized_root)
    ):
        raise PreflightError("invalid_artifact_namespace")
    if output.is_symlink():
        raise PreflightError("output_symlink_collision")
    parent = output.parent.resolve(strict=False)
    if not _is_within(parent, authorized_root):
        raise PreflightError("output_symlink_escape")
    return {
        "logical_output_directory": str(output.absolute()),
        "physical_output_directory": str(physical),
        "physical_artifact_root": str(authorized_root),
        "physical_parent": str(parent),
    }


def _revalidate_output_identity(
    output: Path, identity: dict[str, Any], repo_root: Path
) -> None:
    try:
        current = _resolved_output_identity(output, repo_root)
    except PreflightError as exc:
        raise PreflightError("output_identity_mismatch") from exc
    for key in (
        "physical_output_directory",
        "physical_artifact_root",
        "physical_parent",
    ):
        if current.get(key) != identity.get(key):
            raise PreflightError("output_identity_mismatch")
    if Path(str(identity.get("logical_output_directory", ""))).resolve(
        strict=False
    ) != Path(current["physical_output_directory"]):
        raise PreflightError("output_identity_mismatch")


def _bind_physical_output(
    output: Path, identity: dict[str, Any], repo_root: Path
) -> Path:
    _revalidate_output_identity(output, identity, repo_root)
    return Path(identity["physical_output_directory"])


def _runtime_digest(value: dict[str, Any]) -> str:
    return sha256_canonical(
        {key: item for key, item in value.items() if key != "state_digest"}
    )


def _attempt_digest(value: dict[str, Any]) -> str:
    return sha256_canonical(
        {key: item for key, item in value.items() if key != "record_digest"}
    )


def _runtime_ledger_files(run_dir: Path) -> list[Path]:
    root = run_dir / "runtime-ledger"
    return sorted(root.glob("generation-*.json")) if root.is_dir() else []


def _attempt_event_files(run_dir: Path, task_id: str) -> list[Path]:
    root = run_dir / "attempt-ledger" / task_id
    return sorted(root.glob("event-*.json")) if root.is_dir() else []


def _attempt_heads_digest(heads: dict[str, str]) -> str:
    return sha256_canonical({key: heads[key] for key in sorted(heads)})


def _validate_attempt_ledgers(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    allow_running: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    expected_tasks = manifest.get("expected_tasks")
    if not isinstance(expected_tasks, list):
        raise PreflightError("attempt_ledger_invalid")
    expected_ids = [item.get("task_id") for item in expected_tasks]
    ledger_root = run_dir / "attempt-ledger"
    if ledger_root.exists() and any(
        item.is_dir() and item.name not in set(expected_ids)
        for item in ledger_root.iterdir()
    ):
        raise PreflightError("attempt_ledger_integrity_failure")
    histories: dict[str, list[dict[str, Any]]] = {}
    heads: dict[str, str] = {}
    binding = {
        "run_id": manifest.get("run_id"),
        "mode": manifest.get("mode"),
        "plan_digest": manifest.get("plan_digest"),
        "authorization_digest": manifest.get("authorization_digest"),
        "proposal_digest": manifest.get("proposal_digest"),
        "target_environment_digest": manifest.get("target_environment_digest"),
    }
    for task_id in expected_ids:
        files = _attempt_event_files(run_dir, str(task_id))
        events: list[dict[str, Any]] = []
        previous: str | None = None
        attempt_states: dict[int, str] = {}
        successful = 0
        for index, path in enumerate(files, 1):
            if path.name != f"event-{index:06d}.json":
                raise PreflightError("attempt_ledger_integrity_failure")
            event = _strict_json_file(path)
            required = {
                "artifact_type",
                "schema_version",
                "canonical_task_id",
                "event_number",
                "attempt_number",
                "attempt_identity",
                "run_id",
                "mode",
                "plan_digest",
                "authorization_digest",
                "proposal_digest",
                "target_environment_digest",
                "started_at_utc",
                "completed_at_utc",
                "disposition",
                "reason_code",
                "telemetry_digest",
                "failure_evidence_digest",
                "success_evidence_digest",
                "previous_attempt_digest",
                "record_digest",
            }
            if (
                set(event) != required
                or event.get("artifact_type") != ATTEMPT_LEDGER_ARTIFACT_TYPE
                or event.get("schema_version") != ATTEMPT_LEDGER_SCHEMA_VERSION
                or event.get("canonical_task_id") != task_id
                or event.get("event_number") != index
                or event.get("previous_attempt_digest") != previous
                or event.get("record_digest") != _attempt_digest(event)
                or event.get("disposition") not in ATTEMPT_DISPOSITIONS
                or any(event.get(key) != value for key, value in binding.items())
            ):
                raise PreflightError("attempt_ledger_integrity_failure")
            attempt = event.get("attempt_number")
            if (
                not isinstance(attempt, int)
                or isinstance(attempt, bool)
                or attempt not in (1, 2)
            ):
                raise PreflightError("attempt_ledger_invalid")
            identity = sha256_canonical(
                {"run_id": manifest["run_id"], "task_id": task_id, "attempt": attempt}
            )
            if event.get("attempt_identity") != identity:
                raise PreflightError("attempt_ledger_integrity_failure")
            prior = attempt_states.get(attempt)
            disposition = event["disposition"]
            if prior is None and disposition != "running":
                raise PreflightError("attempt_ledger_invalid")
            if prior is not None and (prior != "running" or disposition == "running"):
                raise PreflightError("attempt_ledger_invalid")
            if disposition == "running" and index == len(files) and not allow_running:
                raise PreflightError("attempt_ledger_invalid")
            if disposition in {
                "interrupted",
                "transient_failure",
                "permanent_failure",
            }:
                evidence_path = (
                    run_dir
                    / "attempt-failures"
                    / str(task_id)
                    / f"attempt-{attempt}.json"
                )
                evidence = _strict_json_file(evidence_path)
                if sha256_canonical(evidence) != event.get("failure_evidence_digest"):
                    raise PreflightError("attempt_ledger_integrity_failure")
                quarantine_path = evidence.get("quarantine_path")
                telemetry_digest = event.get("telemetry_digest")
                if telemetry_digest is not None:
                    telemetry_path = (
                        run_dir / str(quarantine_path) / "telemetry.json"
                        if isinstance(quarantine_path, str)
                        else None
                    )
                    if (
                        telemetry_path is None
                        or sha256_canonical(_strict_json_file(telemetry_path))
                        != telemetry_digest
                    ):
                        raise PreflightError("attempt_ledger_integrity_failure")
            if disposition == "succeeded":
                result = _strict_json_file(
                    run_dir / "fits" / str(task_id) / "result.json"
                )
                telemetry = _strict_json_file(
                    run_dir
                    / "fits"
                    / str(task_id)
                    / "attempts"
                    / f"attempt-{attempt}"
                    / "telemetry.json"
                )
                if sha256_canonical(result) != event.get(
                    "success_evidence_digest"
                ) or sha256_canonical(telemetry) != event.get("telemetry_digest"):
                    raise PreflightError("attempt_ledger_integrity_failure")
            attempt_states[attempt] = disposition
            successful += int(disposition == "succeeded")
            previous = event["record_digest"]
            events.append(event)
        if successful > 1:
            raise PreflightError("duplicate_successful_attempt")
        if 2 in attempt_states and attempt_states.get(1) not in {
            "transient_failure",
            "interrupted",
        }:
            raise PreflightError("attempt_ledger_invalid")
        histories[str(task_id)] = events
        if previous is not None:
            heads[str(task_id)] = previous
    return histories, heads


def _append_attempt_event(
    run_dir: Path,
    manifest: dict[str, Any],
    task_id: str,
    attempt: int,
    disposition: str,
    *,
    started_at_utc: str,
    completed_at_utc: str | None = None,
    reason_code: str | None = None,
    telemetry_digest: str | None = None,
    failure_evidence_digest: str | None = None,
    success_evidence_digest: str | None = None,
) -> dict[str, Any]:
    if disposition not in ATTEMPT_DISPOSITIONS:
        raise PreflightError("attempt_ledger_invalid")
    histories, _ = _validate_attempt_ledgers(run_dir, manifest, allow_running=True)
    events = histories[task_id]
    if disposition == "running":
        terminal = events[-1]["disposition"] if events else None
        maximum_attempt = max((item["attempt_number"] for item in events), default=0)
        if (
            attempt != maximum_attempt + 1
            or attempt > 2
            or (attempt == 2 and terminal not in {"transient_failure", "interrupted"})
        ):
            raise PreflightError("retry_limit_exceeded")
    elif (
        not events
        or events[-1]["disposition"] != "running"
        or events[-1]["attempt_number"] != attempt
    ):
        raise PreflightError("attempt_ledger_invalid")
    event = {
        "artifact_type": ATTEMPT_LEDGER_ARTIFACT_TYPE,
        "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
        "canonical_task_id": task_id,
        "event_number": len(events) + 1,
        "attempt_number": attempt,
        "attempt_identity": sha256_canonical(
            {"run_id": manifest["run_id"], "task_id": task_id, "attempt": attempt}
        ),
        "run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "plan_digest": manifest["plan_digest"],
        "authorization_digest": manifest.get("authorization_digest"),
        "proposal_digest": manifest.get("proposal_digest"),
        "target_environment_digest": manifest.get("target_environment_digest"),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "disposition": disposition,
        "reason_code": reason_code,
        "telemetry_digest": telemetry_digest,
        "failure_evidence_digest": failure_evidence_digest,
        "success_evidence_digest": success_evidence_digest,
        "previous_attempt_digest": events[-1]["record_digest"] if events else None,
    }
    event["record_digest"] = _attempt_digest(event)
    _atomic_create_json(
        run_dir / "attempt-ledger" / task_id / f"event-{len(events) + 1:06d}.json",
        event,
    )
    return event


def _validate_runtime_ledger(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    wall_now: datetime | None = None,
    validate_attempt_heads: bool = True,
) -> dict[str, Any]:
    files = _runtime_ledger_files(run_dir)
    if not files:
        raise PreflightError("runtime_ledger_invalid")
    previous_digest: str | None = None
    previous_elapsed = -1.0
    previous_failure_count = -1
    original_started: str | None = None
    head: dict[str, Any] | None = None
    expected_binding = {
        "run_id": manifest.get("run_id"),
        "normalized_output_directory": manifest.get("physical_output_directory"),
        "authorization_digest": manifest.get("authorization_digest"),
        "target_environment_digest": manifest.get("target_environment_digest"),
        "proposal_digest": manifest.get("proposal_digest"),
        "resource_policy_digest": manifest.get("resource_policy_digest"),
        "plan_digest": manifest.get("plan_digest"),
        "machine_profile_digest": manifest.get("machine_profile_digest"),
    }
    for generation, path in enumerate(files):
        if path.name != f"generation-{generation:06d}.json":
            raise PreflightError("runtime_ledger_integrity_failure")
        value = _strict_json_file(path)
        required = {
            "artifact_type",
            "schema_version",
            "run_id",
            "normalized_output_directory",
            "authorization_digest",
            "target_environment_digest",
            "proposal_digest",
            "resource_policy_digest",
            "plan_digest",
            "machine_profile_digest",
            "original_started_at_utc",
            "generation",
            "previous_generation_digest",
            "accumulated_elapsed_seconds",
            "last_accounted_utc",
            "failure_count",
            "attempt_ledger_heads_digest",
            "state_digest",
        }
        elapsed = value.get("accumulated_elapsed_seconds")
        failure_count = value.get("failure_count")
        if (
            set(value) != required
            or value.get("artifact_type") != RUNTIME_LEDGER_ARTIFACT_TYPE
            or value.get("schema_version") != RUNTIME_LEDGER_SCHEMA_VERSION
            or value.get("generation") != generation
            or value.get("previous_generation_digest") != previous_digest
            or value.get("state_digest") != _runtime_digest(value)
            or any(
                value.get(key) != expected for key, expected in expected_binding.items()
            )
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < previous_elapsed
            or not isinstance(failure_count, int)
            or isinstance(failure_count, bool)
            or failure_count < previous_failure_count
        ):
            raise PreflightError("runtime_ledger_integrity_failure")
        if generation == 0 and value.get("previous_generation_digest") is not None:
            raise PreflightError("runtime_ledger_integrity_failure")
        if original_started is None:
            original_started = value.get("original_started_at_utc")
        elif value.get("original_started_at_utc") != original_started:
            raise PreflightError("runtime_ledger_integrity_failure")
        started = _parse_utc(value["original_started_at_utc"])
        accounted = _parse_utc(value["last_accounted_utc"])
        if accounted < started:
            raise PreflightError("runtime_ledger_invalid")
        if (
            float(elapsed) + RUNTIME_CLOCK_SKEW_SECONDS
            < (accounted - started).total_seconds()
        ):
            raise PreflightError("runtime_ledger_invalid")
        previous_elapsed = float(elapsed)
        previous_failure_count = failure_count
        previous_digest = value["state_digest"]
        head = value
    checkpoint = manifest.get("runtime_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or head is None
        or checkpoint
        != {
            "generation": head["generation"],
            "state_digest": head["state_digest"],
            "accumulated_elapsed_seconds": head["accumulated_elapsed_seconds"],
            "attempt_ledger_heads_digest": head["attempt_ledger_heads_digest"],
        }
    ):
        raise PreflightError("runtime_ledger_integrity_failure")
    if validate_attempt_heads:
        _, heads = _validate_attempt_ledgers(run_dir, manifest, allow_running=True)
        if head["attempt_ledger_heads_digest"] != _attempt_heads_digest(heads):
            raise PreflightError("runtime_ledger_integrity_failure")
    now = (wall_now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    last = _parse_utc(head["last_accounted_utc"])
    if now < last - timedelta(seconds=RUNTIME_CLOCK_SKEW_SECONDS):
        raise PreflightError("runtime_clock_rollback")
    return head


def _append_runtime_generation(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    elapsed: float,
    failure_count: int,
    wall_now: datetime | None = None,
) -> dict[str, Any]:
    files = _runtime_ledger_files(run_dir)
    prior = (
        _validate_runtime_ledger(
            run_dir,
            manifest,
            wall_now=wall_now,
            validate_attempt_heads=False,
        )
        if files
        else None
    )
    _, heads = _validate_attempt_ledgers(run_dir, manifest, allow_running=True)
    now = (wall_now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    generation = 0 if prior is None else int(prior["generation"]) + 1
    original_started = (
        now.isoformat().replace("+00:00", "Z")
        if prior is None
        else prior["original_started_at_utc"]
    )
    high_water = max(
        0.0 if prior is None else float(prior["accumulated_elapsed_seconds"]), elapsed
    )
    value = {
        "artifact_type": RUNTIME_LEDGER_ARTIFACT_TYPE,
        "schema_version": RUNTIME_LEDGER_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "normalized_output_directory": manifest["physical_output_directory"],
        "authorization_digest": manifest.get("authorization_digest"),
        "target_environment_digest": manifest.get("target_environment_digest"),
        "proposal_digest": manifest.get("proposal_digest"),
        "resource_policy_digest": manifest.get("resource_policy_digest"),
        "plan_digest": manifest["plan_digest"],
        "machine_profile_digest": manifest["machine_profile_digest"],
        "original_started_at_utc": original_started,
        "generation": generation,
        "previous_generation_digest": None if prior is None else prior["state_digest"],
        "accumulated_elapsed_seconds": high_water,
        "last_accounted_utc": now.isoformat().replace("+00:00", "Z"),
        "failure_count": failure_count,
        "attempt_ledger_heads_digest": _attempt_heads_digest(heads),
    }
    value["state_digest"] = _runtime_digest(value)
    _atomic_create_json(
        run_dir / "runtime-ledger" / f"generation-{generation:06d}.json", value
    )
    manifest["runtime_checkpoint"] = {
        "generation": generation,
        "state_digest": value["state_digest"],
        "accumulated_elapsed_seconds": high_water,
        "attempt_ledger_heads_digest": value["attempt_ledger_heads_digest"],
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    return value


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
    isolate_process_group()
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
    mode: str,
    fixture: bool,
    bounded_authorized: bool,
    repo_root: Path,
    target_environment: dict[str, Any] | None = None,
    authorization_proposal: dict[str, Any] | None = None,
    effective_authorization: dict[str, Any] | None = None,
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
        if not all(
            isinstance(value, dict)
            for value in (
                target_environment,
                authorization_proposal,
                effective_authorization,
            )
        ):
            codes.append("typed_target_authorization_missing")
            if bounded_authorized:
                codes.append("boolean_target_authorization_rejected")
        else:
            head = subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={repo_root.as_posix()}",
                    "-C",
                    str(repo_root),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
                encoding="utf-8",
            ).strip()
            environment_report = validate_target_environment(
                target_environment,
                plan,
                profile,
                expected_source_sha=head,
            )
            proposal_report = validate_authorization_proposal(
                authorization_proposal,
                target_environment,
                plan,
                profile,
            )
            authorization_report = validate_effective_authorization(
                effective_authorization,
                authorization_proposal,
                target_environment,
                plan,
                profile,
            )
            for report in (
                environment_report,
                proposal_report,
                authorization_report,
            ):
                codes.extend(report["reason_codes"])
            if target_environment.get("mode") != mode:
                codes.append("authorization_mode_mismatch")
            try:
                authorized_physical = Path(
                    str(target_environment.get("normalized_output_directory", ""))
                ).resolve(strict=False)
            except (OSError, ValueError):
                authorized_physical = Path()
            if authorized_physical != output_dir.resolve(strict=False):
                codes.append("output_identity_mismatch")
            if (
                target_environment.get("execution_stage")
                != TARGET_INNER_EXECUTION_STAGE
            ):
                codes.append("execution_stage_mismatch")
    expected = (repo_root / ARTIFACT_ROOT).resolve(strict=False)
    try:
        output_identity = _resolved_output_identity(output_dir, repo_root)
    except PreflightError as exc:
        codes.append(str(exc))
        output_identity = None
    if (
        output_identity is None
        or Path(output_identity["physical_output_directory"]).parent != expected
    ):
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


def _terminate_and_reap(processes: list[Any]) -> bool:
    return terminate_and_reap(processes)


def _terminate_tree(process) -> bool:
    return _terminate_and_reap([process])


def _aggregate_process_tree_rss(root_pid: int | None = None) -> int:
    """Return an exact deduplicated sample or fail closed on uncertain races."""
    pid = os.getpid() if root_pid is None else root_pid
    try:
        root = psutil.Process(pid)
        processes = {item.pid: item for item in [root, *root.children(recursive=True)]}
    except (psutil.Error, OSError) as exc:
        raise PreflightError("memory_sampler_failure") from exc
    total = 0
    for sampled_pid, process in processes.items():
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
            # Benign only when a second OS-level existence check proves that the
            # enumerated PID disappeared during this sample.
            if psutil.pid_exists(sampled_pid):
                raise PreflightError("memory_sampler_failure") from exc
        except (psutil.AccessDenied, OSError, psutil.Error) as exc:
            raise PreflightError("memory_sampler_failure") from exc
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


def _directory_size(path: Path) -> int:
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError as exc:
        raise PreflightError("artifact_size_guard_triggered") from exc


def _target_attempt_temp(run_dir: Path, task_id: str, attempt: int) -> Path:
    temporary = _target_attempt_temp_path(run_dir, task_id, attempt)
    if temporary.exists() or temporary.is_symlink():
        raise PreflightError("attempt_temporary_collision")
    temporary.mkdir(parents=True, exist_ok=False)
    return temporary


def _target_attempt_temp_path(run_dir: Path, task_id: str, attempt: int) -> Path:
    identity = sha256_canonical(
        {"run_id": run_dir.name, "task_id": task_id, "attempt": attempt}
    )
    return run_dir / "temporary" / task_id / f"attempt-{attempt}-{identity}.tmp"


def _quarantine_attempt(
    run_dir: Path, temporary: Path, task_id: str, attempt: int, reason: str
) -> Path | None:
    if not temporary.exists():
        return None
    destination = (
        run_dir
        / "quarantine"
        / task_id
        / f"attempt-{attempt}-{reason}-{uuid4().hex[:8]}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    return destination


def _reconcile_interrupted_attempts(
    run_dir: Path,
    manifest: dict[str, Any],
    expected: list[dict[str, Any]],
    runtime_state: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    histories, _ = _validate_attempt_ledgers(run_dir, manifest, allow_running=True)
    interrupted_count = 0
    for task in expected:
        events = histories[task["task_id"]]
        if not events or events[-1]["disposition"] != "running":
            continue
        running = events[-1]
        attempt = running["attempt_number"]
        fit = run_dir / "fits" / task["task_id"]
        if fit.exists() or fit.is_symlink():
            raise PreflightError("canonical_promotion_without_attempt_commit")
        temporary = _target_attempt_temp_path(run_dir, task["task_id"], attempt)
        telemetry_digest = None
        if (temporary / "telemetry.json").is_file():
            telemetry_digest = sha256_canonical(
                _strict_json_file(temporary / "telemetry.json")
            )
        quarantine = _quarantine_attempt(
            run_dir,
            temporary,
            task["task_id"],
            attempt,
            "interrupted_on_resume",
        )
        evidence = {
            "task_id": task["task_id"],
            "attempt": attempt,
            "reason_code": "interrupted_on_resume",
            "record_digest": telemetry_digest,
            "quarantine_path": str(quarantine.relative_to(run_dir))
            if quarantine is not None
            else None,
        }
        _atomic_create_json(
            run_dir / "attempt-failures" / task["task_id"] / f"attempt-{attempt}.json",
            evidence,
        )
        _append_attempt_event(
            run_dir,
            manifest,
            task["task_id"],
            attempt,
            "interrupted",
            started_at_utc=running["started_at_utc"],
            completed_at_utc=_utc(),
            reason_code="interrupted_on_resume",
            telemetry_digest=telemetry_digest,
            failure_evidence_digest=sha256_canonical(evidence),
        )
        interrupted_count += 1
    if not interrupted_count:
        return runtime_state
    failure_count = int(runtime_state["failure_count"]) + interrupted_count
    return _append_runtime_generation(
        run_dir,
        manifest,
        elapsed=elapsed,
        failure_count=failure_count,
    )


def _prepromotion_guards(
    *,
    run_dir: Path,
    temporary: Path,
    task: dict[str, Any],
    attempt: int,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
    target_environment: dict[str, Any],
    authorization_proposal: dict[str, Any],
    effective_authorization: dict[str, Any],
    control_values: dict[str, dict[str, Any]],
    repo_root: Path,
    elapsed: float,
    timed_out: bool,
    rss_sampler: Callable[[int | None], int] = _aggregate_process_tree_rss,
    disk_usage_provider: Callable[[Path], Any] = shutil.disk_usage,
) -> None:
    report = validate_effective_authorization(
        effective_authorization,
        authorization_proposal,
        target_environment,
        plan,
        profile,
    )
    if not report["valid"]:
        raise PreflightError(",".join(report["reason_codes"]))
    if timed_out:
        raise PreflightError("fit_timeout")
    if elapsed >= plan["limits"]["global_wall_clock_seconds"]["value"]:
        raise PreflightError("global_wall_clock_exceeded")
    if rss_sampler(None) > plan["limits"]["aggregate_process_tree_rss_bytes"]["value"]:
        raise PreflightError("aggregate_memory_guard_triggered")
    try:
        available_ram = psutil.virtual_memory().available
    except (OSError, psutil.Error, AttributeError) as exc:
        raise PreflightError("memory_sampler_failure") from exc
    if available_ram < plan["limits"]["minimum_system_available_ram_bytes"]["value"]:
        raise PreflightError("insufficient_available_ram")
    _revalidate_output_identity(run_dir, manifest["output_identity"], repo_root)
    _revalidate_control_files(manifest["control_files"], control_values)
    try:
        free = disk_usage_provider(run_dir).free
    except (OSError, AttributeError) as exc:
        raise PreflightError("live_disk_lookup_failed") from exc
    if free < plan["limits"]["minimum_free_disk_bytes"]["value"]:
        raise PreflightError("insufficient_free_disk")
    projected = (
        _directory_size(run_dir) + _directory_size(temporary) + PROMOTION_OVERHEAD_BYTES
    )
    if projected > plan["limits"]["artifact_size_bytes"]["value"]:
        raise PreflightError("artifact_size_guard_triggered")
    histories, _ = _validate_attempt_ledgers(run_dir, manifest, allow_running=True)
    events = histories[task["task_id"]]
    if (
        not events
        or events[-1].get("attempt_number") != attempt
        or events[-1].get("disposition") != "running"
    ):
        raise PreflightError("attempt_ledger_integrity_failure")
    _validate_runtime_ledger(run_dir, manifest)


def _promote_target_attempt(
    run_dir: Path,
    temporary: Path,
    task: dict[str, Any],
    record: dict[str, Any],
) -> None:
    task_id = task["task_id"]
    attempt = int(record["attempt"])
    if record.get("task_id") != task_id:
        raise PreflightError("retry_identity_mismatch")
    fit = run_dir / "fits" / task_id
    destination = fit / "attempts" / f"attempt-{attempt}"
    if fit.exists() or fit.is_symlink() or destination.exists():
        raise PreflightError("promotion_collision")
    destination.parent.mkdir(parents=True, exist_ok=False)
    os.replace(temporary, destination)
    _atomic_create_json(fit / "result.json", record)
    _atomic_create_json(
        fit / "COMPLETED.json",
        {
            "task_id": task_id,
            "attempt": attempt,
            "record_digest": sha256_canonical(record),
        },
    )


def _finalization_guards(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
    target_environment: dict[str, Any],
    authorization_proposal: dict[str, Any],
    effective_authorization: dict[str, Any],
    control_values: dict[str, dict[str, Any]],
    repo_root: Path,
    elapsed: float,
) -> None:
    report = validate_effective_authorization(
        effective_authorization,
        authorization_proposal,
        target_environment,
        plan,
        profile,
    )
    if not report["valid"]:
        raise PreflightError(",".join(report["reason_codes"]))
    if elapsed >= plan["limits"]["global_wall_clock_seconds"]["value"]:
        raise PreflightError("global_wall_clock_exceeded")
    _revalidate_control_files(manifest["control_files"], control_values)
    _revalidate_output_identity(run_dir, manifest["output_identity"], repo_root)
    try:
        free_disk = shutil.disk_usage(run_dir).free
    except (OSError, AttributeError) as exc:
        raise PreflightError("live_disk_lookup_failed") from exc
    if free_disk < plan["limits"]["minimum_free_disk_bytes"]["value"]:
        raise PreflightError("insufficient_free_disk")
    try:
        available_ram = psutil.virtual_memory().available
    except (OSError, psutil.Error, AttributeError) as exc:
        raise PreflightError("memory_sampler_failure") from exc
    if available_ram < plan["limits"]["minimum_system_available_ram_bytes"]["value"]:
        raise PreflightError("insufficient_available_ram")
    if (
        _aggregate_process_tree_rss()
        > plan["limits"]["aggregate_process_tree_rss_bytes"]["value"]
    ):
        raise PreflightError("aggregate_memory_guard_triggered")
    if (
        _directory_size(run_dir) + PROMOTION_OVERHEAD_BYTES
        > plan["limits"]["artifact_size_bytes"]["value"]
    ):
        raise PreflightError("artifact_size_guard_triggered")
    _validate_attempt_ledgers(run_dir, manifest, allow_running=False)
    _validate_runtime_ledger(run_dir, manifest)


def run(
    plan: dict[str, Any],
    profile: dict[str, Any],
    output_dir: Path,
    *,
    mode: str,
    repo_root: Path | None = None,
    fixture: bool = False,
    bounded_authorized: bool = False,
    target_environment: dict[str, Any] | None = None,
    authorization_proposal: dict[str, Any] | None = None,
    effective_authorization: dict[str, Any] | None = None,
    machine_profile_path: Path | None = None,
    target_environment_path: Path | None = None,
    authorization_proposal_path: Path | None = None,
    effective_authorization_path: Path | None = None,
    max_tasks: int | None = None,
    timeout_seconds: float | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    output_dir = Path(output_dir).absolute()
    if not fixture and (max_tasks is not None or timeout_seconds is not None):
        raise PreflightError("target_scope_override_forbidden")
    guard = execution_guard(
        plan=plan,
        profile=profile,
        output_dir=output_dir,
        mode=mode,
        fixture=fixture,
        bounded_authorized=bounded_authorized,
        repo_root=root,
        target_environment=target_environment,
        authorization_proposal=authorization_proposal,
        effective_authorization=effective_authorization,
    )
    if not guard["authorized"]:
        raise PreflightError(",".join(guard["reason_codes"]))
    output_identity = _resolved_output_identity(output_dir, root)
    control_values: dict[str, dict[str, Any]] = {}
    control_files: dict[str, Any] = {}
    if not fixture:
        paths = {
            "machine_profile": machine_profile_path,
            "target_environment": target_environment_path,
            "authorization_proposal": authorization_proposal_path,
            "effective_authorization": effective_authorization_path,
        }
        if any(path is None for path in paths.values()):
            raise PreflightError("canonical_control_path_missing")
        control_values = {
            "machine_profile": profile,
            "target_environment": target_environment,
            "authorization_proposal": authorization_proposal,
            "effective_authorization": effective_authorization,
        }
        artifact_digests = {
            "machine_profile": profile["profile_digest"],
            "target_environment": target_environment["environment_digest"],
            "authorization_proposal": authorization_proposal["proposal_digest"],
            "effective_authorization": effective_authorization["authorization_digest"],
        }
        control_files = {
            name: _control_binding(
                paths[name], control_values[name], artifact_digests[name]
            )
            for name in paths
        }
        # This is the final canonical-source read before the first run mutation.
        _revalidate_control_files(control_files, control_values)
    if output_dir.exists() or output_dir.is_symlink():
        raise PreflightError("artifact_namespace_already_exists")
    physical_output = Path(output_identity["physical_output_directory"])
    physical_output.parent.mkdir(parents=True, exist_ok=True)
    physical_output.mkdir()
    _revalidate_output_identity(output_dir, output_identity, root)
    # All mutations use the already-resolved physical namespace. The logical
    # spelling remains evidence only and is re-resolved at every later gate.
    output_dir = _bind_physical_output(output_dir, output_identity, root)
    expected = _tasks(plan, mode, max_tasks)
    _atomic_json(output_dir / "plan.json", plan)
    _atomic_json(output_dir / "machine_profile.json", profile)
    if not fixture:
        _atomic_json(output_dir / "target_environment.json", target_environment)
        _atomic_json(output_dir / "authorization_proposal.json", authorization_proposal)
        _atomic_json(
            output_dir / "effective_authorization.json", effective_authorization
        )
    manifest = {
        "schema_version": 1 if fixture else 2,
        "run_id": output_dir.name,
        "normalized_output_directory": str(output_dir.resolve(strict=False)),
        "physical_output_directory": output_identity["physical_output_directory"],
        "output_identity": output_identity,
        "mode": mode,
        "evidence_scope": EVIDENCE_FIXTURE if fixture else "target_single_vm_measured",
        "scientific_manifest_digest": SCIENTIFIC_DIGEST,
        "plan_digest": plan["plan_digest"],
        "machine_profile_digest": profile["profile_digest"],
        "execution_stage": None if fixture else TARGET_INNER_EXECUTION_STAGE,
        "target_environment_digest": None
        if fixture
        else target_environment["environment_digest"],
        "proposal_digest": None
        if fixture
        else authorization_proposal["proposal_digest"],
        "authorization_digest": None
        if fixture
        else effective_authorization["authorization_digest"],
        "authorization_expires_at": None
        if fixture
        else effective_authorization["expires_at"],
        "resource_policy_digest": None
        if fixture
        else target_environment["resource_policy_digest"],
        "control_files": control_files,
        "max_workers": MODES[mode],
        "threads_per_worker": 2,
        "expected_tasks": expected,
        "created_utc": _utc(),
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)
    if not fixture:
        _append_runtime_generation(output_dir, manifest, elapsed=0.0, failure_count=0)
    return resume(
        plan,
        profile,
        output_dir,
        mode=mode,
        repo_root=root,
        fixture=fixture,
        bounded_authorized=bounded_authorized,
        target_environment=target_environment,
        authorization_proposal=authorization_proposal,
        effective_authorization=effective_authorization,
        machine_profile_path=machine_profile_path,
        target_environment_path=target_environment_path,
        authorization_proposal_path=authorization_proposal_path,
        effective_authorization_path=effective_authorization_path,
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
    target_environment: dict[str, Any] | None = None,
    authorization_proposal: dict[str, Any] | None = None,
    effective_authorization: dict[str, Any] | None = None,
    machine_profile_path: Path | None = None,
    target_environment_path: Path | None = None,
    authorization_proposal_path: Path | None = None,
    effective_authorization_path: Path | None = None,
    max_tasks: int | None = None,
    timeout_seconds: float | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    run_dir = Path(run_dir).absolute()
    if not fixture and (max_tasks is not None or timeout_seconds is not None):
        raise PreflightError("target_scope_override_forbidden")
    guard = execution_guard(
        plan=plan,
        profile=profile,
        output_dir=run_dir,
        mode=mode,
        fixture=fixture,
        bounded_authorized=bounded_authorized,
        repo_root=root,
        target_environment=target_environment,
        authorization_proposal=authorization_proposal,
        effective_authorization=effective_authorization,
    )
    if not guard["authorized"]:
        raise PreflightError(",".join(guard["reason_codes"]))
    manifest = _read(run_dir / "run_manifest.json", [])
    if not fixture:
        persisted_environment = _read(run_dir / "target_environment.json", [])
        persisted_proposal = _read(run_dir / "authorization_proposal.json", [])
        persisted_authorization = _read(run_dir / "effective_authorization.json", [])
        if (
            persisted_environment != target_environment
            or persisted_proposal != authorization_proposal
            or persisted_authorization != effective_authorization
        ):
            raise PreflightError("replacement_authorization_forbidden")
        control_values = {
            "machine_profile": profile,
            "target_environment": target_environment,
            "authorization_proposal": authorization_proposal,
            "effective_authorization": effective_authorization,
        }
        supplied_paths = {
            "machine_profile": machine_profile_path,
            "target_environment": target_environment_path,
            "authorization_proposal": authorization_proposal_path,
            "effective_authorization": effective_authorization_path,
        }
        if any(path is None for path in supplied_paths.values()):
            raise PreflightError("canonical_control_path_missing")
        bindings = manifest.get("control_files") if isinstance(manifest, dict) else None
        if not isinstance(bindings, dict) or any(
            str(Path(supplied_paths[name]).absolute())
            != bindings.get(name, {}).get("canonical_path")
            or str(Path(supplied_paths[name]).resolve(strict=True))
            != bindings.get(name, {}).get("source_path")
            for name in supplied_paths
        ):
            raise PreflightError("replacement_authorization_forbidden")
        _revalidate_control_files(bindings, control_values)
        run_dir = _bind_physical_output(run_dir, manifest["output_identity"], root)
        runtime_state = _validate_runtime_ledger(run_dir, manifest)
        last_accounted = _parse_utc(runtime_state["last_accounted_utc"])
        conservative_gap = max(
            0.0, (datetime.now(timezone.utc) - last_accounted).total_seconds()
        )
        accumulated_before_session = (
            runtime_state["accumulated_elapsed_seconds"] + conservative_gap
        )
    else:
        runtime_state = None
        accumulated_before_session = 0.0
    if (
        not manifest
        or manifest.get("plan_digest") != plan["plan_digest"]
        or manifest.get("machine_profile_digest") != profile["profile_digest"]
        or manifest.get("mode") != mode
        or (
            not fixture
            and (
                manifest.get("execution_stage") != TARGET_INNER_EXECUTION_STAGE
                or manifest.get("target_environment_digest")
                != target_environment["environment_digest"]
                or manifest.get("proposal_digest")
                != authorization_proposal["proposal_digest"]
                or manifest.get("authorization_digest")
                != effective_authorization["authorization_digest"]
                or manifest.get("authorization_expires_at")
                != effective_authorization["expires_at"]
            )
        )
        or (
            manifest.get("normalized_output_directory") is not None
            and manifest.get("physical_output_directory")
            != str(run_dir.resolve(strict=False))
        )
    ):
        raise PreflightError("incompatible_resume")
    if (run_dir / "COMPLETED.json").exists():
        completed_report = validate_artifacts(run_dir)
        if not completed_report["valid"]:
            raise PreflightError("completed_run_invalid")
        return {
            "run_id": run_dir.name,
            "executed": 0,
            "skipped": completed_report["completed"],
            "completed": 0,
            "failed": 0,
            "validation": completed_report,
        }
    expected = _tasks(plan, mode, max_tasks)
    if not fixture:
        runtime_state = _reconcile_interrupted_attempts(
            run_dir,
            manifest,
            expected,
            runtime_state,
            accumulated_before_session,
        )
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
    if fixture:
        attempts: dict[str, int] = {}
    else:
        histories, _ = _validate_attempt_ledgers(run_dir, manifest, allow_running=False)
        attempts = {
            task_id: max((event["attempt_number"] for event in events), default=0)
            for task_id, events in histories.items()
        }
    completed = failed = 0
    started_global = time.monotonic()

    def account_runtime(*, persist: bool = True) -> float:
        nonlocal runtime_state
        elapsed = accumulated_before_session + max(
            0.0, time.monotonic() - started_global
        )
        if runtime_state is not None and persist:
            runtime_state = _append_runtime_generation(
                run_dir,
                manifest,
                elapsed=elapsed,
                failure_count=int(runtime_state["failure_count"]),
            )
        return elapsed

    def abort_active(reason_code: str) -> None:
        nonlocal runtime_state
        items = list(active.values())
        cleanup_ok = _terminate_and_reap([item[0] for item in items])
        queue_results = [close_process_queue(item[1]) for item in items]
        queues_ok = all(queue_results)
        cleanup_reason = None if cleanup_ok and queues_ok else "process_cleanup_failure"
        if not fixture:
            for _process, _queue, task, _started, started_utc, temporary in items:
                attempt = attempts[task["task_id"]]
                failure = {
                    "task_id": task["task_id"],
                    "attempt": attempt,
                    "status": "failed",
                    "reason_code": reason_code,
                    "completed_utc": _utc(),
                }
                if cleanup_reason is not None:
                    failure["supervisor_cleanup_reason_code"] = cleanup_reason
                if temporary is not None and temporary.exists():
                    runner_telemetry = temporary / "telemetry.json"
                    if not runner_telemetry.exists():
                        _atomic_create_json(runner_telemetry, failure)
                quarantine = (
                    _quarantine_attempt(
                        run_dir, temporary, task["task_id"], attempt, reason_code
                    )
                    if temporary is not None
                    else None
                )
                evidence = {
                    "task_id": task["task_id"],
                    "attempt": attempt,
                    "reason_code": reason_code,
                    "record_digest": sha256_canonical(failure),
                    "quarantine_path": str(quarantine.relative_to(run_dir))
                    if quarantine is not None
                    else None,
                }
                if cleanup_reason is not None:
                    evidence["supervisor_cleanup_reason_code"] = cleanup_reason
                _atomic_create_json(
                    run_dir
                    / "attempt-failures"
                    / task["task_id"]
                    / f"attempt-{attempt}.json",
                    evidence,
                )
                _append_attempt_event(
                    run_dir,
                    manifest,
                    task["task_id"],
                    attempt,
                    "permanent_failure",
                    started_at_utc=started_utc,
                    completed_at_utc=failure["completed_utc"],
                    reason_code=reason_code,
                    telemetry_digest=sha256_canonical(failure),
                    failure_evidence_digest=sha256_canonical(evidence),
                )
            runtime_state["failure_count"] = int(runtime_state["failure_count"]) + len(
                items
            )
            account_runtime()
        active.clear()
        if cleanup_reason is not None:
            raise PreflightError(f"{reason_code};{cleanup_reason}")
        raise PreflightError(reason_code)

    peak_aggregate = 0
    while pending or active:
        if not fixture:
            try:
                _revalidate_control_files(manifest["control_files"], control_values)
                _revalidate_output_identity(run_dir, manifest["output_identity"], root)
                if (
                    shutil.disk_usage(run_dir).free
                    < plan["limits"]["minimum_free_disk_bytes"]["value"]
                ):
                    raise PreflightError("insufficient_free_disk")
            except PreflightError as exc:
                abort_active(str(exc))
            except OSError:
                abort_active("live_disk_lookup_failed")
            current_authorization = validate_effective_authorization(
                effective_authorization,
                authorization_proposal,
                target_environment,
                plan,
                profile,
            )
            if not current_authorization["valid"]:
                abort_active(",".join(current_authorization["reason_codes"]))
        if (
            account_runtime(persist=False)
            > plan["limits"]["global_wall_clock_seconds"]["value"]
        ):
            abort_active("global_wall_clock_exceeded")
        while pending and len(active) < workers:
            task = pending.pop(0)
            attempts[task["task_id"]] = attempts.get(task["task_id"], 0) + 1
            if attempts[task["task_id"]] > retry_max + 1:
                raise PreflightError("retry_limit_exceeded")
            if not fixture:
                try:
                    _revalidate_control_files(manifest["control_files"], control_values)
                    _revalidate_output_identity(
                        run_dir, manifest["output_identity"], root
                    )
                    disk = shutil.disk_usage(run_dir)
                    if disk.free < plan["limits"]["minimum_free_disk_bytes"]["value"]:
                        raise PreflightError("insufficient_free_disk")
                except PreflightError as exc:
                    abort_active(str(exc))
                except OSError:
                    abort_active("live_disk_lookup_failed")
                started_attempt_utc = _utc()
                _append_attempt_event(
                    run_dir,
                    manifest,
                    task["task_id"],
                    attempts[task["task_id"]],
                    "running",
                    started_at_utc=started_attempt_utc,
                )
                account_runtime()
            else:
                started_attempt_utc = _utc()
            temporary = (
                None
                if fixture
                else _target_attempt_temp(
                    run_dir, task["task_id"], attempts[task["task_id"]]
                )
            )
            q = context.Queue()
            p = context.Process(target=_worker, args=(q, task, str(root), fixture))
            p.start()
            active[task["task_id"]] = (
                p,
                q,
                task,
                time.monotonic(),
                started_attempt_utc,
                temporary,
            )
        try:
            sampled_rss = _aggregate_process_tree_rss()
        except PreflightError:
            abort_active("memory_sampler_failure")
        peak_aggregate = max(peak_aggregate, sampled_rss)
        try:
            available_ram = psutil.virtual_memory().available
        except (OSError, psutil.Error, AttributeError):
            abort_active("memory_sampler_failure")
        if (
            peak_aggregate > plan["limits"]["aggregate_process_tree_rss_bytes"]["value"]
            or available_ram
            < plan["limits"]["minimum_system_available_ram_bytes"]["value"]
        ):
            abort_active("aggregate_memory_guard_triggered")
        finished = []
        for task_id, (p, q, task, started, started_utc, temporary) in active.items():
            if not p.is_alive() or time.monotonic() - started >= timeout:
                finished.append(task_id)
        if not finished:
            time.sleep(0.01)
            continue
        for task_id in finished:
            p, q, task, started, started_utc, temporary = active.pop(task_id)
            timed_out = p.is_alive()
            cleaned = (
                _terminate_tree(p)
                if timed_out
                else (p.join(1) is None and not p.is_alive())
            )
            try:
                # Queue.empty() is explicitly unreliable across processes: a
                # successful spawned worker can exit before its feeder thread
                # makes telemetry visible to the parent.
                msg = q.get(timeout=1)
            except Empty:
                msg = {
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
                "target_environment_digest": manifest.get("target_environment_digest"),
                "proposal_digest": manifest.get("proposal_digest"),
                "authorization_digest": manifest.get("authorization_digest"),
                "orphan_cleanup_pass": cleaned,
            }
            if fixture:
                _promote_attempt(run_dir, task, record)
                artifact_bytes = _directory_size(run_dir)
                if artifact_bytes > plan["limits"]["artifact_size_bytes"]["value"]:
                    raise PreflightError("artifact_size_guard_triggered")
            else:
                assert temporary is not None
                _atomic_create_json(temporary / "telemetry.json", record)
                telemetry_digest = sha256_canonical(record)
                if reason is None:
                    try:
                        _prepromotion_guards(
                            run_dir=run_dir,
                            temporary=temporary,
                            task=task,
                            attempt=attempts[task_id],
                            manifest=manifest,
                            plan=plan,
                            profile=profile,
                            target_environment=target_environment,
                            authorization_proposal=authorization_proposal,
                            effective_authorization=effective_authorization,
                            control_values=control_values,
                            repo_root=root,
                            elapsed=account_runtime(persist=False),
                            timed_out=timed_out,
                        )
                        _promote_target_attempt(run_dir, temporary, task, record)
                    except PreflightError as exc:
                        reason = str(exc).split(",")[0]
                        record["status"] = "failed"
                        record["reason_code"] = reason
                if reason is None:
                    success_digest = sha256_canonical(record)
                    _append_attempt_event(
                        run_dir,
                        manifest,
                        task_id,
                        attempts[task_id],
                        "succeeded",
                        started_at_utc=started_utc,
                        completed_at_utc=record["completed_utc"],
                        telemetry_digest=telemetry_digest,
                        success_evidence_digest=success_digest,
                    )
                else:
                    quarantine = _quarantine_attempt(
                        run_dir, temporary, task_id, attempts[task_id], reason
                    )
                    evidence = {
                        "task_id": task_id,
                        "attempt": attempts[task_id],
                        "reason_code": reason,
                        "record_digest": sha256_canonical(record),
                        "quarantine_path": str(quarantine.relative_to(run_dir))
                        if quarantine is not None
                        else None,
                    }
                    failure_path = (
                        run_dir
                        / "attempt-failures"
                        / task_id
                        / f"attempt-{attempts[task_id]}.json"
                    )
                    _atomic_create_json(failure_path, evidence)
                    _append_attempt_event(
                        run_dir,
                        manifest,
                        task_id,
                        attempts[task_id],
                        "transient_failure" if transient else "permanent_failure",
                        started_at_utc=started_utc,
                        completed_at_utc=record["completed_utc"],
                        reason_code=reason,
                        telemetry_digest=telemetry_digest,
                        failure_evidence_digest=sha256_canonical(evidence),
                    )
                    runtime_state["failure_count"] = (
                        int(runtime_state["failure_count"]) + 1
                    )
                account_runtime()
            if reason is None:
                completed += 1
            elif transient and attempts[task_id] <= retry_max:
                pending.insert(0, task)
            else:
                failed += 1
                if fail_fast:
                    if not fixture and active:
                        abort_active("fail_fast_interrupted")
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
    if not fixture:
        # Final runtime/attempt state and its manifest mirror are durable before
        # private validation. No artifact is mutated after public validation.
        final_elapsed = account_runtime()
        _finalization_guards(
            run_dir=run_dir,
            manifest=manifest,
            plan=plan,
            profile=profile,
            target_environment=target_environment,
            authorization_proposal=authorization_proposal,
            effective_authorization=effective_authorization,
            control_values=control_values,
            repo_root=root,
            elapsed=final_elapsed,
        )
    report = validate_artifacts(
        run_dir, allow_incomplete=failed > 0 or len(records) < len(expected)
    )
    _atomic_json(run_dir / "validation_report.json", report)
    if report["valid"] and len(records) == len(expected):
        marker = {
            "run_id": run_dir.name,
            "validation_report_digest": sha256_canonical(report),
        }
        if fixture:
            _atomic_json(run_dir / "COMPLETED.json", marker)
        else:
            _atomic_create_json(run_dir / "COMPLETED.json", marker)
            report = validate_artifacts(run_dir)
            if not report["valid"]:
                raise PreflightError("post_marker_validation_failure")
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
    typed_target = (
        manifest.get("evidence_scope") == "target_single_vm_measured"
        and manifest.get("schema_version") == 2
    )
    environment = proposal = authorization = None
    if typed_target:
        environment = _read(run_dir / "target_environment.json", codes)
        proposal = _read(run_dir / "authorization_proposal.json", codes)
        authorization = _read(run_dir / "effective_authorization.json", codes)
        if not all(
            isinstance(value, dict) for value in (environment, proposal, authorization)
        ):
            codes.append("typed_target_authorization_missing")
        else:
            for report in (
                validate_target_environment(environment, plan, profile),
                validate_authorization_proposal(proposal, environment, plan, profile),
                validate_effective_authorization(
                    authorization,
                    proposal,
                    environment,
                    plan,
                    profile,
                    enforce_current_expiry=False,
                ),
            ):
                codes.extend(report["reason_codes"])
            if (
                manifest.get("execution_stage") != TARGET_INNER_EXECUTION_STAGE
                or manifest.get("target_environment_digest")
                != environment.get("environment_digest")
                or manifest.get("proposal_digest") != proposal.get("proposal_digest")
                or manifest.get("authorization_digest")
                != authorization.get("authorization_digest")
                or manifest.get("authorization_expires_at")
                != authorization.get("expires_at")
            ):
                codes.append("authorization_provenance_mismatch")
            try:
                _revalidate_control_files(
                    manifest.get("control_files", {}),
                    {
                        "machine_profile": profile,
                        "target_environment": environment,
                        "authorization_proposal": proposal,
                        "effective_authorization": authorization,
                    },
                )
            except PreflightError:
                codes.append("control_file_mismatch")
            try:
                attempt_histories, _attempt_heads = _validate_attempt_ledgers(
                    run_dir, manifest, allow_running=False
                )
                if any(
                    events and events[-1]["disposition"] != "succeeded"
                    for events in attempt_histories.values()
                ):
                    codes.append("failed_unit_present")
            except PreflightError as exc:
                codes.append(str(exc))
            try:
                _validate_runtime_ledger(run_dir, manifest)
            except PreflightError as exc:
                codes.append(str(exc))
    if manifest.get("run_id") != run_dir.name:
        codes.append("foreign_machine_or_run")
    if manifest.get("physical_output_directory") is not None and manifest.get(
        "physical_output_directory"
    ) != str(run_dir.resolve(strict=False)):
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
        if typed_target and (
            result.get("target_environment_digest")
            != manifest.get("target_environment_digest")
            or result.get("proposal_digest") != manifest.get("proposal_digest")
            or result.get("authorization_digest")
            != manifest.get("authorization_digest")
        ):
            codes.append("authorization_provenance_mismatch")
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
            rebuilt = (
                summarize_legacy_v1(records, mode=manifest["mode"])
                if stored.get("schema_version") == 1
                else summarize(records, mode=manifest["mode"])
            )
        except PreflightError:
            rebuilt = None
        if rebuilt is None or not _equivalent_json(stored, rebuilt):
            codes.append("summary_mismatch")
        warmup_count = stored.get(
            "warmups_excluded", stored.get("warmup", {}).get("count", 0)
        )
        if warmup_count and stored.get("measured_count", 0) + warmup_count != len(
            records
        ):
            codes.append("warmup_mixed_into_measured")
    elif stored:
        expected_summary = {
            "mode": manifest["mode"],
            "measured_count": 0,
            "warmups_excluded": sum(
                x.get("classification") == "warmup" for x in records
            ),
        }
        if stored != expected_summary:
            codes.append("summary_mismatch")
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
