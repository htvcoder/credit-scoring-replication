"""Typed authorization contracts for target P7C.4B.2b inner-fit preflight."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from creditrep.config.loader import sha256_canonical
from creditrep.protocols.p7c4b2b import (
    MODES,
    SCIENTIFIC_DIGEST,
    PreflightError,
    machine_profile_digest,
    validate_machine,
    validate_plan,
)

SCHEMA_VERSION = 1
EXECUTION_STAGE = "target_inner_fit_projection_preflight"
ENVIRONMENT_TYPE = "p7c4b2b_target_environment"
PROPOSAL_TYPE = "p7c4b2b_target_authorization_proposal"
AUTHORIZATION_TYPE = "p7c4b2b_target_effective_authorization"
APPROVAL_PHRASE = "APPROVE_P7C4B2B_TARGET_INNER_PREFLIGHT"
SHA40 = re.compile(r"[0-9a-f]{40}")

ENVIRONMENT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "execution_stage",
        "source_git_commit",
        "scientific_manifest_digest",
        "plan_digest",
        "mode",
        "task_ids",
        "task_count",
        "machine_profile_digest",
        "cloud_provider",
        "instance_type",
        "worker_count",
        "normalized_output_directory",
        "captured_at",
        "resource_policy",
        "resource_policy_digest",
        "environment_digest",
    }
)
PROPOSAL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "execution_stage",
        "source_git_commit",
        "scientific_manifest_digest",
        "plan_digest",
        "mode",
        "task_ids",
        "task_count",
        "maximum_task_count",
        "machine_profile_digest",
        "worker_count",
        "normalized_output_directory",
        "run_id",
        "environment_digest",
        "resource_policy_digest",
        "authorization_effective",
        "created_at",
        "proposal_digest",
    }
)
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "execution_stage",
        "operator_identity",
        "operator_approval",
        "created_at",
        "expires_at",
        "source_git_commit",
        "scientific_manifest_digest",
        "plan_digest",
        "environment_digest",
        "proposal_digest",
        "machine_profile_digest",
        "mode",
        "task_ids",
        "task_count",
        "maximum_task_count",
        "worker_count",
        "normalized_output_directory",
        "run_id",
        "resource_policy_digest",
        "authorization_effective",
        "authorization_digest",
    }
)


def _digest(value: dict[str, Any], field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return sha256_canonical(payload)


def environment_digest(value: dict[str, Any]) -> str:
    return _digest(value, "environment_digest")


def proposal_digest(value: dict[str, Any]) -> str:
    return _digest(value, "proposal_digest")


def authorization_digest(value: dict[str, Any]) -> str:
    return _digest(value, "authorization_digest")


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _canonical_tasks(plan: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    validate_plan(plan)
    if mode not in MODES:
        raise PreflightError("authorization_mode_invalid")
    tasks = [task for task in plan["tasks"] if task["mode"] == mode]
    if len(tasks) != 54:
        raise PreflightError("authorization_task_scope_mismatch")
    return tasks


def canonical_task_ids(plan: dict[str, Any], mode: str) -> list[str]:
    return [task["task_id"] for task in _canonical_tasks(plan, mode)]


def resource_policy(plan: dict[str, Any]) -> dict[str, Any]:
    """Return only the already-reviewed B2b limits; do not invent outer limits."""
    validate_plan(plan)
    return {
        "limits": deepcopy(plan["limits"]),
        "thread_policy": deepcopy(plan["thread_policy"]),
    }


def render_target_environment(
    plan: dict[str, Any],
    profile: dict[str, Any],
    *,
    mode: str,
    output_directory: Path,
    captured_at: str,
) -> dict[str, Any]:
    validate_machine(profile, plan)
    tasks = canonical_task_ids(plan, mode)
    policy = resource_policy(plan)
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ENVIRONMENT_TYPE,
        "execution_stage": EXECUTION_STAGE,
        "source_git_commit": profile.get("git_commit"),
        "scientific_manifest_digest": SCIENTIFIC_DIGEST,
        "plan_digest": plan["plan_digest"],
        "mode": mode,
        "task_ids": tasks,
        "task_count": len(tasks),
        "machine_profile_digest": profile["profile_digest"],
        "cloud_provider": profile["cloud_provider"],
        "instance_type": profile["instance_type"],
        "worker_count": MODES[mode],
        "normalized_output_directory": str(output_directory.resolve()),
        "captured_at": captured_at,
        "resource_policy": policy,
        "resource_policy_digest": sha256_canonical(policy),
    }
    value["environment_digest"] = environment_digest(value)
    report = validate_target_environment(value, plan, profile)
    if not report["valid"]:
        raise PreflightError(",".join(report["reason_codes"]))
    return value


def validate_target_environment(
    value: Any,
    plan: dict[str, Any],
    profile: dict[str, Any],
    *,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    codes: list[str] = []
    if not isinstance(value, dict) or set(value) != ENVIRONMENT_FIELDS:
        return {"valid": False, "reason_codes": ["environment_schema_invalid"]}
    try:
        validate_machine(profile, plan)
        expected_tasks = canonical_task_ids(plan, value.get("mode"))
        policy = resource_policy(plan)
    except (PreflightError, KeyError, TypeError):
        expected_tasks, policy = [], {}
        codes.append("environment_scope_invalid")
    source = value.get("source_git_commit")
    if not isinstance(source, str) or SHA40.fullmatch(source) is None:
        codes.append("source_git_commit_invalid")
    if expected_source_sha is not None and source != expected_source_sha:
        codes.append("source_git_commit_mismatch")
    if source != profile.get("git_commit"):
        codes.append("source_git_commit_mismatch")
    if (
        value.get("artifact_type") != ENVIRONMENT_TYPE
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        codes.append("environment_schema_invalid")
    if value.get("execution_stage") != EXECUTION_STAGE:
        codes.append("execution_stage_mismatch")
    if value.get("scientific_manifest_digest") != SCIENTIFIC_DIGEST:
        codes.append("scientific_manifest_digest_mismatch")
    if value.get("plan_digest") != plan.get("plan_digest"):
        codes.append("plan_digest_mismatch")
    if value.get("task_ids") != expected_tasks or value.get("task_count") != 54:
        codes.append("authorization_task_scope_mismatch")
    if value.get("machine_profile_digest") != profile.get(
        "profile_digest"
    ) or profile.get("profile_digest") != machine_profile_digest(profile):
        codes.append("machine_profile_digest_mismatch")
    if value.get("cloud_provider") != profile.get("cloud_provider") or value.get(
        "instance_type"
    ) != profile.get("instance_type"):
        codes.append("machine_profile_identity_mismatch")
    if value.get("worker_count") != MODES.get(value.get("mode")):
        codes.append("worker_count_mismatch")
    output = value.get("normalized_output_directory")
    if not isinstance(output, str) or not output or not Path(output).is_absolute():
        codes.append("output_identity_invalid")
    if _timestamp(value.get("captured_at")) is None:
        codes.append("timestamp_invalid")
    if value.get("resource_policy") != policy or value.get(
        "resource_policy_digest"
    ) != sha256_canonical(policy):
        codes.append("resource_policy_mismatch")
    if value.get("environment_digest") != environment_digest(value):
        codes.append("environment_digest_mismatch")
    return {"valid": not codes, "reason_codes": sorted(set(codes))}


def render_authorization_proposal(
    environment: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
    *,
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    report = validate_target_environment(environment, plan, profile)
    if not report["valid"]:
        raise PreflightError(",".join(report["reason_codes"]))
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise PreflightError("run_identity_invalid")
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PROPOSAL_TYPE,
        "execution_stage": EXECUTION_STAGE,
        "source_git_commit": environment["source_git_commit"],
        "scientific_manifest_digest": environment["scientific_manifest_digest"],
        "plan_digest": environment["plan_digest"],
        "mode": environment["mode"],
        "task_ids": environment["task_ids"],
        "task_count": 54,
        "maximum_task_count": 54,
        "machine_profile_digest": environment["machine_profile_digest"],
        "worker_count": environment["worker_count"],
        "normalized_output_directory": environment["normalized_output_directory"],
        "run_id": run_id,
        "environment_digest": environment["environment_digest"],
        "resource_policy_digest": environment["resource_policy_digest"],
        "authorization_effective": False,
        "created_at": created_at,
    }
    value["proposal_digest"] = proposal_digest(value)
    report = validate_authorization_proposal(value, environment, plan, profile)
    if not report["valid"]:
        raise PreflightError(",".join(report["reason_codes"]))
    return value


def validate_authorization_proposal(
    value: Any,
    environment: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    codes: list[str] = []
    if not isinstance(value, dict) or set(value) != PROPOSAL_FIELDS:
        return {"valid": False, "reason_codes": ["proposal_schema_invalid"]}
    environment_report = validate_target_environment(environment, plan, profile)
    if not environment_report["valid"]:
        codes.append("environment_invalid")
    bindings = {
        "source_git_commit": "source_git_commit_mismatch",
        "scientific_manifest_digest": "scientific_manifest_digest_mismatch",
        "plan_digest": "plan_digest_mismatch",
        "mode": "authorization_mode_mismatch",
        "task_ids": "authorization_task_scope_mismatch",
        "machine_profile_digest": "machine_profile_digest_mismatch",
        "worker_count": "worker_count_mismatch",
        "normalized_output_directory": "output_identity_mismatch",
        "environment_digest": "environment_digest_mismatch",
        "resource_policy_digest": "resource_policy_mismatch",
    }
    for field, code in bindings.items():
        if value.get(field) != environment.get(field):
            codes.append(code)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact_type") != PROPOSAL_TYPE
    ):
        codes.append("proposal_schema_invalid")
    if value.get("execution_stage") != EXECUTION_STAGE:
        codes.append("execution_stage_mismatch")
    if value.get("task_count") != 54 or value.get("maximum_task_count") != 54:
        codes.append("authorization_task_scope_mismatch")
    if value.get("authorization_effective") is not False:
        codes.append("proposal_cannot_authorize")
    if (
        not isinstance(value.get("run_id"), str)
        or Path(value["run_id"]).name != value["run_id"]
    ):
        codes.append("run_identity_invalid")
    if Path(str(value.get("normalized_output_directory"))).name != value.get("run_id"):
        codes.append("run_output_identity_mismatch")
    if _timestamp(value.get("created_at")) is None:
        codes.append("timestamp_invalid")
    if value.get("proposal_digest") != proposal_digest(value):
        codes.append("proposal_digest_mismatch")
    return {"valid": not codes, "reason_codes": sorted(set(codes))}


def create_effective_authorization(
    proposal: dict[str, Any],
    environment: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
    *,
    operator_identity: str,
    operator_approval: str,
    created_at: str,
    expires_at: str,
) -> dict[str, Any]:
    report = validate_authorization_proposal(proposal, environment, plan, profile)
    if not report["valid"]:
        raise PreflightError(",".join(report["reason_codes"]))
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": AUTHORIZATION_TYPE,
        "execution_stage": EXECUTION_STAGE,
        "operator_identity": operator_identity,
        "operator_approval": operator_approval,
        "created_at": created_at,
        "expires_at": expires_at,
        "source_git_commit": proposal["source_git_commit"],
        "scientific_manifest_digest": proposal["scientific_manifest_digest"],
        "plan_digest": proposal["plan_digest"],
        "environment_digest": proposal["environment_digest"],
        "proposal_digest": proposal["proposal_digest"],
        "machine_profile_digest": proposal["machine_profile_digest"],
        "mode": proposal["mode"],
        "task_ids": proposal["task_ids"],
        "task_count": 54,
        "maximum_task_count": 54,
        "worker_count": proposal["worker_count"],
        "normalized_output_directory": proposal["normalized_output_directory"],
        "run_id": proposal["run_id"],
        "resource_policy_digest": proposal["resource_policy_digest"],
        "authorization_effective": True,
    }
    value["authorization_digest"] = authorization_digest(value)
    report = validate_effective_authorization(
        value, proposal, environment, plan, profile, now=None
    )
    if not report["valid"]:
        raise PreflightError(",".join(report["reason_codes"]))
    return value


def validate_effective_authorization(
    value: Any,
    proposal: dict[str, Any],
    environment: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
    *,
    now: datetime | None = None,
    enforce_current_expiry: bool = True,
) -> dict[str, Any]:
    codes: list[str] = []
    if not isinstance(value, dict) or set(value) != AUTHORIZATION_FIELDS:
        return {"valid": False, "reason_codes": ["authorization_schema_invalid"]}
    proposal_report = validate_authorization_proposal(
        proposal, environment, plan, profile
    )
    if not proposal_report["valid"]:
        codes.append("proposal_invalid")
    for field in (
        "source_git_commit",
        "scientific_manifest_digest",
        "plan_digest",
        "environment_digest",
        "proposal_digest",
        "machine_profile_digest",
        "mode",
        "task_ids",
        "task_count",
        "maximum_task_count",
        "worker_count",
        "normalized_output_directory",
        "run_id",
        "resource_policy_digest",
    ):
        if value.get(field) != proposal.get(field):
            codes.append("authorization_scope_mismatch")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact_type") != AUTHORIZATION_TYPE
    ):
        codes.append("authorization_schema_invalid")
    if value.get("execution_stage") != EXECUTION_STAGE:
        codes.append("execution_stage_mismatch")
    if (
        not isinstance(value.get("operator_identity"), str)
        or not value["operator_identity"].strip()
    ):
        codes.append("operator_identity_missing")
    if value.get("operator_approval") != APPROVAL_PHRASE:
        codes.append("operator_approval_invalid")
    if value.get("authorization_effective") is not True:
        codes.append("authorization_not_effective")
    created = _timestamp(value.get("created_at"))
    expires = _timestamp(value.get("expires_at"))
    if created is None or expires is None or expires <= created:
        codes.append("authorization_expiry_invalid")
    elif enforce_current_expiry and (now or datetime.now(timezone.utc)) >= expires:
        codes.append("authorization_expired")
    if value.get("authorization_digest") != authorization_digest(value):
        codes.append("authorization_digest_mismatch")
    return {"valid": not codes, "reason_codes": sorted(set(codes))}
