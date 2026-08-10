from __future__ import annotations

import json

from creditrep.experiments.p7c4b2d_cli import main as cli_main
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2c import build_plan
from creditrep.protocols.p7c4b2d import (
    ENVIRONMENT_FIELDS,
    decision_package,
    environment_digest,
    estimate_cost,
    render_authorization_proposal,
    select_canary,
    task_inventory,
    validate_authorization_proposal,
    validate_target_environment,
)


def _plan():
    return build_plan(
        load_manifest("configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml")
    )


def _environment(plan):
    value = {field: "fixture" for field in ENVIRONMENT_FIELDS}
    value.update(
        {
            "vcpu_count": 2,
            "ram_bytes": 8,
            "gpu_count": 0,
            "gpu_vram_bytes": 0,
            "free_disk_bytes": 100,
            "worker_count": 1,
            "vm_count": 1,
            "execution_mode": "cpu_parallel_1",
            "git_commit": "fixture-head",
            "expected_git_commit": "fixture-head",
            "dataset_hashes": {"AC": "fixture"},
            "plan_digest": plan["plan_digest"],
            "hourly_price": 2.5,
            "maximum_runtime_hours": 10.0,
            "maximum_monetary_budget": 25.0,
            "process_spawn_supported": True,
        }
    )
    value["environment_digest"] = environment_digest(value)
    return value


def test_inventory_derives_all_counts_and_ordering_from_plan():
    inventory = task_inventory(_plan())
    assert inventory["models"] == ["mlp_1", "mlp_3", "mlp_5"]
    assert inventory["proxy_classes"] == [
        "high_cost_proxy",
        "low_cost_proxy",
        "typical_proxy",
    ]
    assert inventory["execution_modes"] == ["cpu_parallel_1", "cpu_parallel_2"]
    assert inventory["datasets"] == ["AC", "GC", "GMC", "HMEQ", "TC", "TH02"]
    assert inventory["mode_qualified_proxy_representatives"] == 18
    assert inventory["strata"] == 108
    assert inventory["warmup_tasks"] == 108
    assert inventory["measured_tasks"] == 216
    assert inventory["total_tasks"] == inventory["upper_bound_tasks"] == 324
    assert inventory["duplicate_sample_id_count"] == 0
    assert inventory["deterministic_order"] is True
    assert inventory["warmup_repetition_ids"] == [0]
    assert inventory["measured_repetition_ids"] == [0, 1]


def test_environment_contract_fails_closed_for_unknown_and_mismatch():
    plan = _plan()
    report = validate_target_environment({}, plan)
    assert "missing_target_environment_metadata" in report["reason_codes"]
    assert "git_provenance_mismatch" in report["reason_codes"]
    environment = _environment(plan)
    assert validate_target_environment(environment, plan)["valid"] is True
    environment["worker_count"] = 2
    assert (
        "worker_count_mismatch"
        in validate_target_environment(environment, plan)["reason_codes"]
    )
    environment = _environment(plan)
    environment["execution_mode"] = "unsupported"
    assert (
        "execution_mode_unsupported"
        in validate_target_environment(environment, plan)["reason_codes"]
    )
    environment = _environment(plan)
    environment["vm_count"] = None
    assert (
        "missing_target_environment_metadata"
        in validate_target_environment(environment, plan)["reason_codes"]
    )


def test_canary_is_deterministic_plan_subset_and_non_scientific():
    first = select_canary(_plan(), "cpu_parallel_2")
    second = select_canary(_plan(), "cpu_parallel_2")
    assert first == second
    assert first["execution_stage"] == "target_canary"
    assert first["scientific_projection_eligible"] is False
    assert first["task_count"] == 4


def test_cost_is_unknown_without_price_and_two_vm_billing_is_explicit():
    hours = {"lower": 1.0, "central": 2.0, "upper": 3.0, "compute_hours": 4.0}
    unknown = estimate_cost(
        mode="cpu_parallel_1", wall_clock_hours=hours, hourly_price=None
    )
    assert unknown["status"] == "unknown_price_input"
    assert unknown["estimated_cost_central"] is None
    priced = estimate_cost(
        mode="cpu_parallel_2",
        wall_clock_hours=hours,
        hourly_price=5.0,
        currency="USD",
        price_source="operator",
        price_observed_at="2026-08-10",
        vm_count=2,
    )
    assert priced["estimated_cost_lower"] == 10.0
    assert priced["estimated_cost_central"] == 20.0
    assert priced["estimated_cost_upper"] == 30.0
    single_vm = estimate_cost(
        mode="cpu_parallel_2",
        wall_clock_hours=hours,
        hourly_price=5.0,
        currency="USD",
        price_source="operator",
        price_observed_at="2026-08-10",
    )
    assert single_vm["vm_count"] == 1
    assert single_vm["estimated_cost_central"] == 10.0


def test_proposal_is_non_effective_and_digest_bound():
    plan = _plan()
    environment = _environment(plan)
    proposal = render_authorization_proposal(
        plan,
        environment,
        execution_stage="target_canary",
        task_ids=[plan["tasks"][0]["sample_id"]],
        expiry=None,
    )
    assert proposal["authorization_effective"] is False
    assert validate_authorization_proposal(proposal, plan, environment)["valid"] is True
    proposal["plan_digest"] = "wrong"
    assert (
        "authorization_proposal_digest_mismatch"
        in validate_authorization_proposal(proposal, plan, environment)["reason_codes"]
    )


def test_decision_package_fails_closed_then_reaches_canary_review_only():
    plan = _plan()
    blocked = decision_package(plan)
    assert blocked["readiness"] == "NOT_READY_FOR_AUTHORIZATION"
    assert "missing_target_environment_metadata" in blocked["reason_codes"]
    assert "incomplete_canary" in blocked["reason_codes"]
    pricing_and_approval_missing = _environment(plan)
    pricing_and_approval_missing.update(
        {
            "hourly_price": None,
            "maximum_runtime_hours": None,
            "maximum_monetary_budget": None,
        }
    )
    pricing_and_approval_missing["environment_digest"] = environment_digest(
        pricing_and_approval_missing
    )
    missing_review_inputs = decision_package(
        plan,
        pricing_and_approval_missing,
        canary_complete=True,
        canary_approved=True,
    )
    assert "price_input_missing" in missing_review_inputs["reason_codes"]
    assert "operator_approval_missing" in missing_review_inputs["reason_codes"]
    assert "target_canary_not_approved" in blocked["reason_codes"]
    ready = decision_package(
        plan, _environment(plan), canary_complete=True, canary_approved=True
    )
    assert ready["readiness"] == "READY_FOR_CANARY_AUTHORIZATION_REVIEW"
    assert ready["execution_plan_eligible"] is False


def test_safe_cli_only_renders_review_output(tmp_path, capsys):
    assert cli_main(["review-plan"]) == 3
    plan = _plan()
    environment = _environment(plan)
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(environment), encoding="utf-8")
    assert cli_main(["inspect-target-requirements", "--environment", str(path)]) == 0
    assert cli_main(["render-authorization-proposal", "--environment", str(path)]) == 0
    assert "authorization_effective" in capsys.readouterr().out
