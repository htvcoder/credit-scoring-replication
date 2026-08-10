"""Fail-closed target-preflight review and authorization-readiness contracts."""

from __future__ import annotations

import math
from typing import Any

from creditrep.protocols.p7c4b2c import (
    DATASET_OUTER_REPEATS,
    MODES,
    PROXIES,
    canonical_digest,
    validate_plan,
)

SCHEMA_VERSION = 1
CHECKPOINT = "P7C.4B.2d"
ENVIRONMENT_FIELDS = (
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
    "worker_count",
    "execution_mode",
    "vm_count",
    "git_commit",
    "environment_lock_hash",
    "dataset_hashes",
    "output_directory",
    "hourly_price",
    "currency",
    "price_source",
    "price_observed_at",
    "maximum_runtime_hours",
    "maximum_monetary_budget",
    "authorization_identity",
    "authorization_scope",
    "authorization_timestamp",
    "authorization_expiry",
)
PRE_RUN_CODES = {
    "git_provenance_mismatch",
    "plan_digest_mismatch",
    "dataset_input_hash_mismatch",
    "insufficient_free_disk",
    "unsupported_process_spawn",
    "missing_target_environment_metadata",
    "output_collision",
    "worker_count_mismatch",
    "execution_mode_unsupported",
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
    return canonical_digest(environment, "environment_digest")


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def task_inventory(plan: dict[str, Any]) -> dict[str, Any]:
    """Code-derived immutable inventory; 18 includes the execution-mode axis."""
    validate_plan(plan)
    tasks = plan["tasks"]
    models = sorted({task["model_id"] for task in tasks})
    proxies = sorted({task["candidate_proxy"] for task in tasks})
    modes = sorted({task["mode"] for task in tasks})
    datasets = sorted({task["dataset_id"] for task in tasks})
    expected_strata = {
        (task["dataset_id"], task["model_id"], task["candidate_proxy"], task["mode"])
        for task in tasks
    }
    warmups = [task for task in tasks if task["classification"] == "warmup"]
    measured = [task for task in tasks if task["classification"] == "measured"]
    identities = [task["sample_id"] for task in tasks]
    return {
        "models": models,
        "proxy_classes": proxies,
        "execution_modes": modes,
        "datasets": datasets,
        "mode_qualified_proxy_representatives": len(models) * len(proxies) * len(modes),
        "strata": len(expected_strata),
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
    """The emitted plan order is classification then deterministic stratum order."""
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


def validate_target_environment(
    environment: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    codes = []
    missing = [field for field in ENVIRONMENT_FIELDS if environment.get(field) is None]
    if missing:
        codes.append("missing_target_environment_metadata")
    if environment.get("environment_digest") != environment_digest(environment):
        codes.append("environment_digest_mismatch")
    if (
        not environment.get("git_commit")
        or not environment.get("expected_git_commit")
        or environment.get("git_commit") != environment.get("expected_git_commit")
    ):
        codes.append("git_provenance_mismatch")
    if environment.get("plan_digest") != plan["plan_digest"]:
        codes.append("plan_digest_mismatch")
    if environment.get("execution_mode") not in MODES:
        codes.append("execution_mode_unsupported")
    elif environment.get("worker_count") != MODES[environment["execution_mode"]]:
        codes.append("worker_count_mismatch")
    if not _finite_positive(environment.get("free_disk_bytes")):
        codes.append("insufficient_free_disk")
    if not _finite_positive(environment.get("vm_count")):
        codes.append("missing_target_environment_metadata")
    if environment.get("process_spawn_supported") is not True:
        codes.append("unsupported_process_spawn")
    if not isinstance(environment.get("dataset_hashes"), dict) or not environment.get(
        "dataset_hashes"
    ):
        codes.append("dataset_input_hash_mismatch")
    return {
        "valid": not codes,
        "reason_codes": sorted(set(codes)),
        "missing_fields": missing,
    }


def select_canary(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    """Immutable-plan subset: AC/light and GMC/heavy, each warmup+measured."""
    if mode not in MODES:
        raise P7C4B2DError("execution_mode_unsupported")
    required = (("AC", "low_cost_proxy"), ("GMC", "high_cost_proxy"))
    selected = []
    for dataset, proxy in required:
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
    ids = [x["sample_id"] for x in selected]
    return {
        "execution_stage": "target_canary",
        "scientific_projection_eligible": False,
        "mode": mode,
        "task_ids": ids,
        "task_count": len(ids),
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
    if mode not in MODES or vm_count < 1:
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
        or not price_observed_at
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
    task_ids: list[str],
    expiry: str | None,
) -> dict[str, Any]:
    """A review object, explicitly unusable as an effective authorization."""
    if execution_stage not in {
        "target_canary",
        "partial_target_preflight",
        "full_target_preflight",
    }:
        raise P7C4B2DError("invalid_execution_stage")
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "authorization_proposal",
        "authorization_effective": False,
        "checkpoint": CHECKPOINT,
        "plan_digest": plan["plan_digest"],
        "git_commit": environment.get("expected_git_commit"),
        "target_environment_digest": environment.get("environment_digest"),
        "execution_stage": execution_stage,
        "execution_mode": environment.get("execution_mode"),
        "vm_count": environment.get("vm_count"),
        "task_ids": task_ids,
        "maximum_task_count": len(task_ids),
        "maximum_runtime_hours": environment.get("maximum_runtime_hours"),
        "maximum_monetary_budget": environment.get("maximum_monetary_budget"),
        "output_directory": environment.get("output_directory"),
        "stop_conditions": sorted(
            PRE_RUN_CODES | RUNTIME_STOP_CODES | AUTHORIZATION_CODES
        ),
        "expiry": expiry,
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
    return {
        "valid": not codes,
        "reason_codes": sorted(codes),
        "authorization_effective": False,
    }


def decision_package(
    plan: dict[str, Any],
    environment: dict[str, Any] | None = None,
    *,
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
        environment_report = validate_target_environment(environment, plan)
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
            "cpu_parallel_1": select_canary(plan, "cpu_parallel_1"),
            "cpu_parallel_2": select_canary(plan, "cpu_parallel_2"),
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
