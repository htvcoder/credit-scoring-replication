from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from creditrep.protocols.p7c import (
    P7CInventoryError,
    load_protocol_inventory,
    validate_protocol_inventory,
)


INVENTORY = Path("configs/protocols/p7c/p7c_protocol_inventory.yaml")


def payload():
    return yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))


def test_inventory_happy_path():
    assert [item["model_id"] for item in load_protocol_inventory(INVENTORY)["models"]][
        3
    ] == "cart"


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
            "exactly 12",
        ),
        (
            lambda value: value["models"][1].__setitem__(
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
