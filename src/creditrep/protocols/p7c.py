"""Validation for the P7C protocol-inventory planning contract; it never trains."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

from creditrep.datasets.registry import find_repo_root
from creditrep.protocols.p7a import load_manifest, manifest_hash


class P7CInventoryError(ValueError):
    """P7C inventory violates a planning/readiness invariant."""


MODEL_IDS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "cart",
    "mlp_1",
    "mlp_3",
    "mlp_5",
    "catboost",
    "tabnet",
    "ft_transformer",
)
ROLES = {"replication_baseline", "main_results_extension", "optional_extension"}
SEARCH_STATUSES = {
    "locked",
    "no_tuning_pending_final_contract",
    "reference_unlocked",
    "decision_pending",
    "feasibility_required",
}
BACKENDS = {"cpu", "cpu_or_gpu", "gpu_recommended"}
FEASIBILITY = {"engineering_evidenced", "not_assessed", "feasibility_required"}
BUDGETS = {"locked", "unresolved"}


def _error(message: str) -> None:
    raise P7CInventoryError(message)


def _relative_reference(
    value: Any, root: Path, field: str, *, required: bool = False
) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not value:
        _error(f"{field} must be a non-empty relative path or null.")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        _error(f"{field} must not be an absolute path.")
    if not (root / value).is_file():
        _error(f"{field} does not exist: {value}")


def validate_protocol_inventory(
    payload: Any, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        _error("schema_version must be 1.")
    models = payload.get("models")
    if not isinstance(models, list):
        _error("models must be a list.")
    ids = [item.get("model_id") for item in models if isinstance(item, dict)]
    if tuple(ids) != MODEL_IDS or len(set(ids)) != len(ids):
        _error(
            f"models must contain the required unique IDs in order: {list(MODEL_IDS)}"
        )
    p7a = load_manifest(root / "configs/protocols/p7a/p7a_candidate_manifest.yaml")
    if payload.get("p7a_manifest_sha256") != manifest_hash(p7a):
        _error("p7a_manifest_sha256 must bind the unchanged P7A manifest.")
    for item in models:
        required = {
            "model_id",
            "display_name",
            "study_role",
            "source_protocol",
            "tuning_required",
            "search_space_status",
            "search_space_reference",
            "final_manifest_reference",
            "decision_record_reference",
            "candidate_count",
            "compute_backend",
            "feasibility_status",
            "compute_budget_status",
            "scientific_execution_status",
            "blockers",
            "next_checkpoint",
        }
        missing = required - set(item)
        if missing:
            _error(f"{item.get('model_id')}: missing fields {sorted(missing)}")
        if (
            item["study_role"] not in ROLES
            or item["search_space_status"] not in SEARCH_STATUSES
        ):
            _error(f"{item['model_id']}: invalid role or search-space status.")
        if (
            not isinstance(item["tuning_required"], bool)
            or item["compute_backend"] not in BACKENDS
        ):
            _error(f"{item['model_id']}: invalid tuning/backend value.")
        if (
            item["feasibility_status"] not in FEASIBILITY
            or item["compute_budget_status"] not in BUDGETS
        ):
            _error(f"{item['model_id']}: invalid feasibility or budget status.")
        if item["scientific_execution_status"] != "not_started":
            _error(f"{item['model_id']}: scientific execution must remain not_started.")
        if item["candidate_count"] is not None and (
            not isinstance(item["candidate_count"], int) or item["candidate_count"] < 0
        ):
            _error(f"{item['model_id']}: candidate_count must be non-negative or null.")
        if not isinstance(item["blockers"], list):
            _error(f"{item['model_id']}: blockers must be a list.")
        for field in (
            "search_space_reference",
            "final_manifest_reference",
            "decision_record_reference",
        ):
            _relative_reference(item[field], root, f"{item['model_id']}.{field}")
    cart = models[3]
    if cart["search_space_status"] != "locked" or cart["candidate_count"] != 12:
        _error("cart must be locked with exactly 12 candidates.")
    _relative_reference(
        cart["final_manifest_reference"],
        root,
        "cart.final_manifest_reference",
        required=True,
    )
    _relative_reference(
        cart["decision_record_reference"],
        root,
        "cart.decision_record_reference",
        required=True,
    )
    if cart["blockers"]:
        _error("locked cart must not retain unresolved blockers.")
    for item in models:
        if item["model_id"] != "cart" and item["search_space_status"] == "locked":
            _error(f"{item['model_id']}: only CART is locked in P7C.1.")
        if (
            item["search_space_status"] != "locked"
            and item["compute_budget_status"] == "locked"
        ):
            _error(
                f"{item['model_id']}: unresolved search space cannot have a locked budget."
            )
    return deepcopy(payload)


def load_protocol_inventory(
    path: str | Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise P7CInventoryError(f"cannot read inventory: {exc}") from exc
    return validate_protocol_inventory(payload, repo_root=repo_root)
