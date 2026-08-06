from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from creditrep.protocols.p7c import (
    P7CInventoryError,
    load_mlp_feasibility_plan,
    load_rf_xgboost_final_manifest,
    load_protocol_inventory,
    validate_mlp_feasibility_plan,
    validate_rf_xgboost_final_manifest,
    validate_protocol_inventory,
)


INVENTORY = Path("configs/protocols/p7c/p7c_protocol_inventory.yaml")
FINAL_MANIFEST = Path("configs/protocols/p7c/p7c_rf_xgboost_final_manifest.yaml")
MLP_FEASIBILITY_PLAN = Path("configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml")


def payload():
    return yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))


def final_payload():
    return yaml.safe_load(FINAL_MANIFEST.read_text(encoding="utf-8"))


def mlp_plan_payload():
    return yaml.safe_load(MLP_FEASIBILITY_PLAN.read_text(encoding="utf-8"))


def test_inventory_happy_path():
    models = load_protocol_inventory(INVENTORY)["models"]
    assert [item["model_id"] for item in models][3] == "cart"
    assert models[1]["search_space_status"] == models[2]["search_space_status"] == "locked"
    assert models[1]["candidate_count"] == 30
    assert models[2]["candidate_count"] == 108


def test_rf_xgboost_final_manifest_happy_path():
    manifest = load_rf_xgboost_final_manifest(FINAL_MANIFEST)
    assert manifest["models"][0]["search_space"]["candidate_count"] == 30
    assert manifest["models"][1]["search_space"]["candidate_count"] == 108


def test_mlp_feasibility_plan_happy_path():
    plan = load_mlp_feasibility_plan(MLP_FEASIBILITY_PLAN)
    assert [item["model_id"] for item in plan["models"]] == ["mlp_1", "mlp_3", "mlp_5"]
    assert plan["expected_fits"] == {"per_model": 20, "total": 60}
    assert plan["execution_approval"]["status"] == "required_before_execution"


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda value: value["models"].pop(), "required unique IDs"),
        (
            lambda value: value["models"].__setitem__(1, value["models"][0]),
            "required unique IDs",
        ),
        (
            lambda value: value["models"][0].__setitem__("study_role", "unknown"),
            "invalid role",
        ),
        (
            lambda value: value["models"][3].__setitem__(
                "final_manifest_reference", None
            ),
            "final_manifest_reference",
        ),
        (
            lambda value: value["models"][3].__setitem__("candidate_count", 11),
            "invalid locked-model candidate count",
        ),
        (
            lambda value: value["models"][4].__setitem__(
                "compute_budget_status", "locked"
            ),
            "unresolved search space",
        ),
        (
            lambda value: value["models"][0].__setitem__(
                "scientific_execution_status", "completed"
            ),
            "not_started",
        ),
        (
            lambda value: value["models"][0].__setitem__(
                "search_space_reference", "C:/absolute/path.yaml"
            ),
            "absolute path",
        ),
    ],
)
def test_inventory_rejects_contract_violations(mutation, match):
    broken = deepcopy(payload())
    mutation(broken)
    with pytest.raises(P7CInventoryError, match=match):
        validate_protocol_inventory(broken)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda value: value["models"][0]["search_space"].__setitem__(
                "candidate_count", 29
            ),
            "exactly preserve",
        ),
        (
            lambda value: value.__setitem__(
                "scientific_execution", {"status": "authorized"}
            ),
            "scientific_execution",
        ),
        (
            lambda value: value["lock"].__setitem__("manifest_sha256", "0" * 64),
            "hash mismatch",
        ),
    ],
)
def test_final_manifest_rejects_contract_violations(mutation, match):
    broken = deepcopy(final_payload())
    mutation(broken)
    with pytest.raises(P7CInventoryError, match=match):
        validate_rf_xgboost_final_manifest(broken)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda value: value["models"][0]["candidates"][0].__setitem__(
                "learning_rate", 0.02
            ),
            "outside P7A reference values",
        ),
        (
            lambda value: value["execution_approval"].__setitem__(
                "unresolved_thresholds", []
            ),
            "thresholds must remain explicit and unresolved",
        ),
        (
            lambda value: value["lock"].__setitem__("plan_sha256", "0" * 64),
            "plan digest mismatch",
        ),
    ],
)
def test_mlp_feasibility_plan_rejects_contract_violations(mutation, match):
    broken = deepcopy(mlp_plan_payload())
    mutation(broken)
    with pytest.raises(P7CInventoryError, match=match):
        validate_mlp_feasibility_plan(broken)
