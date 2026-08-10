"""Fail-closed, static target-preflight evidence and review contracts.

This module never starts a model workload and never creates effective authorization.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import math
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable

import psutil
from creditrep.checksums import get_dataset_checksum
from creditrep.datasets.registry import find_repo_root, get_dataset_spec, load_registry
from creditrep.protocols.p7c4b2c import (
    DATASET_OUTER_REPEATS,
    MODES,
    PROXIES,
    canonical_digest,
    validate_plan,
)

SCHEMA_VERSION = 2
CHECKPOINT = "P7C.4B.2d"
SHA256_RE = re.compile(r"^[A-Fa-f0-9]{64}$")
GIT_RE = re.compile(r"^[a-f0-9]{40}$")
MINIMUM_FREE_DISK_BYTES = 5 * 1024**3
DISK_POLICY = "p7c4b2d-v1-5GiB-static-canary-output-and-quarantine-margin"
LOCK_FILES = ("pyproject.toml", "requirements.txt", "requirements-dev.txt")
ENVIRONMENT_FIELDS = (
    "schema_version",
    "provider",
    "region",
    "instance_id",
    "os",
    "python_version",
    "cpu_model",
    "vcpu_count",
    "ram_bytes",
    "gpu_model",
    "gpu_count",
    "gpu_vram_bytes",
    "disk_type",
    "free_disk_bytes",
    "network_topology",
    "disk_policy",
    "worker_count",
    "execution_mode",
    "vm_count",
    "git_commit",
    "expected_git_commit",
    "plan_digest",
    "environment_lock_hash",
    "dataset_hashes",
    "output_directory",
    "hourly_price",
    "currency",
    "price_source",
    "price_observed_at",
    "maximum_runtime_hours",
    "maximum_monetary_budget",
    "evidence_observed_at",
)
OPERATOR_METADATA_FIELDS = (
    "provider",
    "region",
    "instance_id",
    "disk_type",
    "network_topology",
    "vm_count",
    "hourly_price",
    "currency",
    "price_source",
    "price_observed_at",
    "maximum_runtime_hours",
    "maximum_monetary_budget",
)
OPERATOR_METADATA_PROTECTED_FIELDS = frozenset(ENVIRONMENT_FIELDS) - frozenset(
    OPERATOR_METADATA_FIELDS
)
PRE_RUN_CODES = {
    "git_provenance_mismatch",
    "git_provenance_unknown",
    "plan_digest_mismatch",
    "dataset_input_hash_mismatch",
    "dataset_input_missing",
    "insufficient_free_disk",
    "unsupported_process_spawn",
    "process_spawn_probe_timeout",
    "missing_target_environment_metadata",
    "output_collision",
    "unsafe_output_namespace",
    "worker_count_mismatch",
    "execution_mode_unsupported",
    "environment_lock_mismatch",
    "invalid_environment_value",
}
RUNTIME_STOP_CODES = {
    "memory_limit_exceeded",
    "runtime_budget_exceeded",
    "failure_rate_threshold_exceeded",
    "timing_invariant_failure",
    "artifact_validation_failure",
}
AUTHORIZATION_CODES = {
    "authorization_missing",
    "authorization_mismatch",
    "authorization_expired",
    "operator_approval_missing",
}
SCIENTIFIC_CODES = {
    "price_input_missing",
    "execution_plan_ineligible",
    "target_canary_not_approved",
    "incomplete_canary",
}


class P7C4B2DError(ValueError):
    """Stable review-contract failure."""


def environment_digest(environment: dict[str, Any]) -> str:
    try:
        return canonical_digest(environment, "environment_digest")
    except (TypeError, ValueError):
        # Invalid JSON values (NaN/Infinity, unsupported types) never acquire a digest.
        return ""


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def current_git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    head = result.stdout.strip().lower()
    return head if result.returncode == 0 and GIT_RE.fullmatch(head) else None


def dependency_lock_fingerprint(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    digest = hashlib.sha256()
    files = []
    for relative in LOCK_FILES:
        path = root / relative
        if not path.is_file():
            raise P7C4B2DError("environment_lock_missing")
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        files.append(relative)
    return {
        "algorithm": "sha256-path-nul-bytes-v1",
        "files": files,
        "sha256": digest.hexdigest().upper(),
    }


def required_canary_datasets(plan: dict[str, Any], mode: str) -> tuple[str, ...]:
    return tuple(
        sorted({task["dataset_id"] for task in select_canary(plan, mode)["tasks"]})
    )


def collect_dataset_hashes(
    plan: dict[str, Any], mode: str, *, repo_root: Path | None = None
) -> dict[str, str]:
    root = (repo_root or find_repo_root()).resolve()
    registry = load_registry(repo_root=root)
    values = {}
    for dataset_id in required_canary_datasets(plan, mode):
        spec = get_dataset_spec(dataset_id, registry)
        checksum = get_dataset_checksum(dataset_id, spec.active_file, repo_root=root)
        values[dataset_id] = checksum.actual_sha256
    return values


def probe_process_spawn(timeout_seconds: float = 2.0) -> dict[str, Any]:
    """Bounded child interpreter probe; it does not import or train any model."""
    command = [sys.executable, "-c", "import sys; sys.exit(0)"]
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=timeout_seconds, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "probe": "child_python_exit",
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
        }
    except OSError:
        return {
            "probe": "child_python_exit",
            "status": "error",
            "timeout_seconds": timeout_seconds,
        }
    return {
        "probe": "child_python_exit",
        "status": "pass" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "timeout_seconds": timeout_seconds,
    }


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def merge_operator_metadata(
    collected: dict[str, Any], operator_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Merge the narrow operator-owned envelope and recompute its evidence digest."""
    if not isinstance(operator_metadata, dict):
        raise P7C4B2DError("operator_metadata_invalid")
    keys = set(operator_metadata)
    protected = sorted(keys & OPERATOR_METADATA_PROTECTED_FIELDS)
    unknown = sorted(keys - set(OPERATOR_METADATA_FIELDS))
    if protected:
        raise P7C4B2DError("operator_metadata_canonical_override")
    if unknown:
        raise P7C4B2DError("operator_metadata_unknown_field")
    if keys != set(OPERATOR_METADATA_FIELDS):
        raise P7C4B2DError("operator_metadata_missing_field")
    for field in (
        "provider",
        "region",
        "instance_id",
        "disk_type",
        "network_topology",
        "currency",
        "price_source",
    ):
        if not _nonempty_text(operator_metadata.get(field)):
            raise P7C4B2DError("operator_metadata_invalid")
    if len(operator_metadata["currency"].strip()) != 3 or not _timestamp(
        operator_metadata.get("price_observed_at")
    ):
        raise P7C4B2DError("operator_metadata_invalid")
    for field in (
        "vm_count",
        "hourly_price",
        "maximum_runtime_hours",
        "maximum_monetary_budget",
    ):
        if not _finite_positive(operator_metadata.get(field)):
            raise P7C4B2DError("operator_metadata_invalid")
    merged = {**collected, **operator_metadata}
    merged["environment_digest"] = environment_digest(merged)
    return merged


def collect_target_environment(
    plan: dict[str, Any],
    *,
    mode: str,
    output_directory: str,
    operator_metadata: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Collect verifiable local evidence only; unknown operator inputs remain null."""
    root = (repo_root or find_repo_root()).resolve()
    target = Path(output_directory)
    if not target.is_absolute():
        target = root / target
    try:
        disk = shutil.disk_usage(target if target.exists() else target.parent)
        free_disk = disk.free
    except OSError:
        free_disk = None
    value = {
        "schema_version": SCHEMA_VERSION,
        "provider": None,
        "region": None,
        "instance_id": None,
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor() or platform.uname().processor or None,
        "vcpu_count": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu_model": "none",
        "gpu_count": 0,
        "gpu_vram_bytes": 0,
        "disk_type": None,
        "free_disk_bytes": free_disk,
        "network_topology": None,
        "disk_policy": DISK_POLICY,
        "worker_count": MODES.get(mode),
        "execution_mode": mode,
        "vm_count": None,
        "git_commit": current_git_head(root),
        "expected_git_commit": current_git_head(root),
        "plan_digest": plan.get("plan_digest"),
        "environment_lock_hash": dependency_lock_fingerprint(root)["sha256"],
        "dataset_hashes": None,
        "output_directory": str(target),
        "hourly_price": None,
        "currency": None,
        "price_source": None,
        "price_observed_at": None,
        "maximum_runtime_hours": None,
        "maximum_monetary_budget": None,
        "evidence_observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "process_spawn_probe": probe_process_spawn(),
    }
    try:
        value["dataset_hashes"] = collect_dataset_hashes(plan, mode, repo_root=root)
    except Exception:
        value["dataset_hashes"] = None
    value["environment_digest"] = environment_digest(value)
    return (
        merge_operator_metadata(value, operator_metadata)
        if operator_metadata is not None
        else value
    )


def task_inventory(plan: dict[str, Any]) -> dict[str, Any]:
    validate_plan(plan)
    tasks = plan["tasks"]
    warmups = [task for task in tasks if task["classification"] == "warmup"]
    measured = [task for task in tasks if task["classification"] == "measured"]
    identities = [task["sample_id"] for task in tasks]
    return {
        "models": sorted({task["model_id"] for task in tasks}),
        "proxy_classes": sorted({task["candidate_proxy"] for task in tasks}),
        "execution_modes": sorted({task["mode"] for task in tasks}),
        "datasets": sorted({task["dataset_id"] for task in tasks}),
        "mode_qualified_proxy_representatives": 18,
        "strata": 108,
        "warmup_repetition_ids": sorted({x["repetition"] for x in warmups}),
        "measured_repetition_ids": sorted({x["repetition"] for x in measured}),
        "warmup_tasks": len(warmups),
        "measured_tasks": len(measured),
        "total_tasks": len(tasks),
        "upper_bound_tasks": plan["sampling"]["maximum_tasks"],
        "duplicate_sample_id_count": len(identities) - len(set(identities)),
        "unexpected_strata": [],
        "missing_strata": [],
        "deterministic_order": tasks == build_sorted_task_view(tasks),
        "seed_derivation": "task.seed = sampling.seed + repetition",
        "plan_digest": plan["plan_digest"],
    }


def build_sorted_task_view(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"cpu_parallel_1": 0, "cpu_parallel_2": 1}
    classification = {"warmup": 0, "measured": 1}
    datasets = {name: index for index, name in enumerate(DATASET_OUTER_REPEATS)}
    models = {name: index for index, name in enumerate(("mlp_1", "mlp_3", "mlp_5"))}
    proxies = {name: index for index, name in enumerate(PROXIES)}
    return sorted(
        tasks,
        key=lambda x: (
            order[x["mode"]],
            classification[x["classification"]],
            datasets[x["dataset_id"]],
            models[x["model_id"]],
            proxies[x["candidate_proxy"]],
            x["repetition"],
        ),
    )


def _safe_output(namespace: Any, root: Path) -> tuple[bool, str | None]:
    if not isinstance(namespace, str) or not namespace.strip():
        return False, None
    candidate = Path(namespace).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
        artifact_root = (root / "artifacts").resolve()
        resolved.relative_to(artifact_root)
    except (OSError, ValueError):
        return False, None
    if (
        resolved in {root.resolve(), artifact_root, Path.home().resolve()}
        or resolved == resolved.anchor
    ):
        return False, None
    return True, str(resolved)


def _validate_values(environment: dict[str, Any]) -> list[str]:
    codes = []
    for field in (
        "vcpu_count",
        "ram_bytes",
        "free_disk_bytes",
        "worker_count",
        "vm_count",
        "maximum_runtime_hours",
        "maximum_monetary_budget",
        "hourly_price",
    ):
        if not _finite_positive(environment.get(field)):
            codes.append("invalid_environment_value")
    gpu_count = environment.get("gpu_count")
    gpu_vram = environment.get("gpu_vram_bytes")
    if (
        not isinstance(gpu_count, int)
        or isinstance(gpu_count, bool)
        or gpu_count < 0
        or not isinstance(gpu_vram, int)
        or isinstance(gpu_vram, bool)
        or gpu_vram < 0
        or (
            gpu_count == 0
            and (
                environment.get("gpu_model") not in {None, "none", ""} or gpu_vram != 0
            )
        )
    ):
        codes.append("invalid_environment_value")
    if (
        not isinstance(environment.get("currency"), str)
        or len(environment["currency"].strip()) != 3
    ):
        codes.append("invalid_environment_value")
    if not all(
        _timestamp(environment.get(field))
        for field in ("price_observed_at", "evidence_observed_at")
    ):
        codes.append("invalid_environment_value")
    return codes


def validate_target_environment(
    environment: dict[str, Any],
    plan: dict[str, Any],
    *,
    repo_root: Path | None = None,
    spawn_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read-only Stage 0 validator against local canonical sources, not declarations."""
    root = (repo_root or find_repo_root()).resolve()
    codes: list[str] = []
    if not isinstance(environment, dict):
        environment = {}
    missing = [field for field in ENVIRONMENT_FIELDS if environment.get(field) is None]
    if missing:
        codes.append("missing_target_environment_metadata")
    if environment.get("schema_version") != SCHEMA_VERSION or environment.get(
        "environment_digest"
    ) != environment_digest(environment):
        codes.append("invalid_environment_value")
    actual_head = current_git_head(root)
    if actual_head is None:
        codes.append("git_provenance_unknown")
    elif (
        environment.get("git_commit") != actual_head
        or environment.get("expected_git_commit") != actual_head
    ):
        codes.append("git_provenance_mismatch")
    if environment.get("plan_digest") != plan.get("plan_digest") or not _sha256(
        environment.get("plan_digest")
    ):
        codes.append("plan_digest_mismatch")
    try:
        expected_lock = dependency_lock_fingerprint(root)["sha256"]
        if environment.get("environment_lock_hash") != expected_lock:
            codes.append("environment_lock_mismatch")
    except P7C4B2DError:
        codes.append("environment_lock_mismatch")
    mode = environment.get("execution_mode")
    if mode not in MODES:
        codes.append("execution_mode_unsupported")
    elif environment.get("worker_count") != MODES[mode]:
        codes.append("worker_count_mismatch")
    codes.extend(_validate_values(environment))
    available = environment.get("free_disk_bytes")
    if (
        environment.get("disk_policy") != DISK_POLICY
        or not _finite_positive(available)
        or available < MINIMUM_FREE_DISK_BYTES
    ):
        codes.append("insufficient_free_disk")
    safe, normalized_output = _safe_output(environment.get("output_directory"), root)
    if not safe:
        codes.append("unsafe_output_namespace")
    elif Path(normalized_output).exists() and any(Path(normalized_output).iterdir()):
        codes.append("output_collision")
    if mode in MODES:
        expected_datasets = set(required_canary_datasets(plan, mode))
        supplied = environment.get("dataset_hashes")
        if (
            not isinstance(supplied, dict)
            or set(supplied) != expected_datasets
            or not all(_sha256(value) for value in supplied.values())
        ):
            codes.append("dataset_input_hash_mismatch")
        else:
            try:
                actual_hashes = collect_dataset_hashes(plan, mode, repo_root=root)
                if {
                    key: value.upper() for key, value in supplied.items()
                } != actual_hashes:
                    codes.append("dataset_input_hash_mismatch")
            except Exception:
                codes.append("dataset_input_missing")
    probe = (spawn_probe or probe_process_spawn)()
    if probe.get("status") == "timeout":
        codes.append("process_spawn_probe_timeout")
    elif probe.get("status") != "pass":
        codes.append("unsupported_process_spawn")
    return {
        "valid": not codes,
        "reason_codes": sorted(set(codes)),
        "missing_fields": missing,
        "git_head": actual_head,
        "required_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "available_free_disk_bytes": available,
        "free_disk_margin_bytes": available - MINIMUM_FREE_DISK_BYTES
        if _finite_positive(available)
        else None,
        "disk_policy": DISK_POLICY,
        "normalized_output_directory": normalized_output,
        "process_spawn_probe": probe,
    }


def select_canary(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise P7C4B2DError("execution_mode_unsupported")
    selected = []
    for dataset, proxy in (("AC", "low_cost_proxy"), ("GMC", "high_cost_proxy")):
        rows = [
            x
            for x in plan["tasks"]
            if x["mode"] == mode
            and x["dataset_id"] == dataset
            and x["candidate_proxy"] == proxy
            and x["model_id"] == "mlp_3"
        ]
        selected.extend(x for x in rows if x["classification"] == "warmup")
        selected.append(
            next(
                x
                for x in rows
                if x["classification"] == "measured" and x["repetition"] == 0
            )
        )
    return {
        "execution_stage": "target_canary",
        "scientific_projection_eligible": False,
        "mode": mode,
        "task_ids": [x["sample_id"] for x in selected],
        "task_count": len(selected),
        "tasks": selected,
        "selection": "small_large_light_heavy_mlp_3_warmup_and_measured",
        "plan_digest": plan["plan_digest"],
    }


def estimate_cost(
    *,
    mode: str,
    wall_clock_hours: dict[str, float] | None,
    hourly_price: float | None,
    currency: str | None = None,
    price_source: str | None = None,
    price_observed_at: str | None = None,
    vm_count: int = 1,
) -> dict[str, Any]:
    if (
        mode not in MODES
        or not isinstance(vm_count, int)
        or isinstance(vm_count, bool)
        or vm_count < 1
    ):
        raise P7C4B2DError("execution_mode_unsupported")
    value = {
        "mode": mode,
        "vm_count": vm_count,
        "estimated_compute_hours": None,
        "estimated_wall_clock_hours_lower": None,
        "estimated_wall_clock_hours_central": None,
        "estimated_wall_clock_hours_upper": None,
        "hourly_price": hourly_price,
        "currency": currency,
        "price_source": price_source,
        "price_observed_at": price_observed_at,
        "estimated_cost_lower": None,
        "estimated_cost_central": None,
        "estimated_cost_upper": None,
    }
    if wall_clock_hours is None:
        return {**value, "status": "unknown_timing_input"}
    value.update(
        {
            "estimated_wall_clock_hours_lower": wall_clock_hours["lower"],
            "estimated_wall_clock_hours_central": wall_clock_hours["central"],
            "estimated_wall_clock_hours_upper": wall_clock_hours["upper"],
            "estimated_compute_hours": wall_clock_hours.get("compute_hours"),
        }
    )
    if hourly_price is None:
        return {**value, "status": "unknown_price_input"}
    if (
        not _finite_positive(hourly_price)
        or not currency
        or not price_source
        or not _timestamp(price_observed_at)
    ):
        raise P7C4B2DError("invalid_price_input")
    return {
        **value,
        "status": "priced",
        "estimated_cost_lower": wall_clock_hours["lower"] * hourly_price * vm_count,
        "estimated_cost_central": wall_clock_hours["central"] * hourly_price * vm_count,
        "estimated_cost_upper": wall_clock_hours["upper"] * hourly_price * vm_count,
    }


def render_authorization_proposal(
    plan: dict[str, Any],
    environment: dict[str, Any],
    *,
    execution_stage: str,
    expiry: str | None,
) -> dict[str, Any]:
    if (
        execution_stage != "target_canary"
        or environment.get("execution_mode") not in MODES
    ):
        raise P7C4B2DError("invalid_execution_stage")
    canary = select_canary(plan, environment["execution_mode"])
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "authorization_proposal",
        "authorization_effective": False,
        "checkpoint": CHECKPOINT,
        "plan_digest": plan["plan_digest"],
        "git_commit": environment.get("expected_git_commit"),
        "target_environment_digest": environment.get("environment_digest"),
        "execution_stage": execution_stage,
        "execution_mode": environment["execution_mode"],
        "vm_count": environment.get("vm_count"),
        "task_ids": canary["task_ids"],
        "maximum_task_count": 4,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "disk_policy": DISK_POLICY,
        "maximum_runtime_hours": environment.get("maximum_runtime_hours"),
        "maximum_monetary_budget": environment.get("maximum_monetary_budget"),
        "hourly_price": environment.get("hourly_price"),
        "currency": environment.get("currency"),
        "output_directory": environment.get("output_directory"),
        "proposal_timestamp": environment.get("evidence_observed_at"),
        "expiry": expiry,
        "stop_conditions": sorted(
            PRE_RUN_CODES | RUNTIME_STOP_CODES | AUTHORIZATION_CODES
        ),
        "cost_acknowledgement": "target_execution_may_incur_cost",
    }
    value["proposal_digest"] = canonical_digest(value, "proposal_digest")
    return value


def validate_authorization_proposal(
    proposal: dict[str, Any], plan: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any]:
    codes = []
    if (
        proposal.get("artifact_type") != "authorization_proposal"
        or proposal.get("authorization_effective") is not False
    ):
        codes.append("authorization_proposal_invalid")
    if proposal.get("proposal_digest") != canonical_digest(proposal, "proposal_digest"):
        codes.append("authorization_proposal_digest_mismatch")
    if proposal.get("plan_digest") != plan.get("plan_digest") or proposal.get(
        "target_environment_digest"
    ) != environment.get("environment_digest"):
        codes.append("authorization_mismatch")
    mode = proposal.get("execution_mode")
    if mode not in MODES:
        codes.append("execution_mode_unsupported")
    else:
        expected = select_canary(plan, mode)["task_ids"]
        if (
            proposal.get("task_ids") != expected
            or len(set(proposal.get("task_ids", []))) != 4
        ):
            codes.append("authorization_proposal_task_scope_mismatch")
    for key in (
        "maximum_runtime_hours",
        "maximum_monetary_budget",
        "hourly_price",
        "currency",
        "output_directory",
        "disk_policy",
    ):
        if proposal.get(key) != environment.get(key):
            codes.append("authorization_mismatch")
    if proposal.get("minimum_free_disk_bytes") != MINIMUM_FREE_DISK_BYTES:
        codes.append("authorization_mismatch")
    expiry = proposal.get("expiry")
    if expiry is not None and (
        not _timestamp(expiry)
        or datetime.fromisoformat(expiry.replace("Z", "+00:00")) <= datetime.now(UTC)
    ):
        codes.append("authorization_expired")
    return {
        "valid": not codes,
        "reason_codes": sorted(set(codes)),
        "authorization_effective": False,
    }


def decision_package(
    plan: dict[str, Any],
    environment: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
    canary_complete: bool = False,
    canary_approved: bool = False,
    price_required: bool = True,
) -> dict[str, Any]:
    inventory = task_inventory(plan)
    codes = []
    environment_report = None
    if environment is None:
        codes.append("missing_target_environment_metadata")
    else:
        environment_report = validate_target_environment(
            environment, plan, repo_root=repo_root
        )
        codes.extend(environment_report["reason_codes"])
        if price_required and environment.get("hourly_price") is None:
            codes.append("price_input_missing")
        if (
            environment.get("maximum_runtime_hours") is None
            or environment.get("maximum_monetary_budget") is None
        ):
            codes.append("operator_approval_missing")
    if not canary_complete:
        codes.append("incomplete_canary")
    if not canary_approved:
        codes.append("target_canary_not_approved")
    readiness = (
        "READY_FOR_CANARY_AUTHORIZATION_REVIEW"
        if not codes
        else "NOT_READY_FOR_AUTHORIZATION"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": CHECKPOINT,
        "plan_digest": plan["plan_digest"],
        "task_inventory": inventory,
        "target_environment": environment_report,
        "canary": {
            mode: {
                key: value
                for key, value in select_canary(plan, mode).items()
                if key != "tasks"
            }
            for mode in MODES
        },
        "stop_conditions": {
            "pre_run": sorted(PRE_RUN_CODES),
            "runtime": sorted(RUNTIME_STOP_CODES),
            "authorization": sorted(AUTHORIZATION_CODES),
            "scientific_eligibility": sorted(SCIENTIFIC_CODES),
        },
        "readiness": readiness,
        "reason_codes": sorted(set(codes)),
        "recommended_next_action": "operator_supply_target_environment_and_budget"
        if codes
        else "request_canary_authorization_review",
        "execution_plan_eligible": False,
    }
