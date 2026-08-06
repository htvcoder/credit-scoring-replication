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
LOCKED_FINAL_MODELS = {"cart": 12, "random_forest": 30, "xgboost": 108}


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


def canonical_final_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the hashable portion of a final P7C model manifest."""
    value = deepcopy(manifest)
    value.pop("lock", None)
    return value


def final_manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash a final P7C manifest with the repository canonical JSON algorithm."""
    from creditrep.config.loader import sha256_canonical

    return sha256_canonical(canonical_final_manifest_payload(manifest))


def validate_rf_xgboost_final_manifest(
    payload: Any, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate the locked full-reference RF/XGBoost P7C.2.3 manifest."""
    root = (repo_root or find_repo_root()).resolve()
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        _error("final_manifest: schema_version must be 1.")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol != {
        "id": "p7c-rf-xgboost-final-scientific-search-space",
        "version": "1.0.0",
        "status": "locked",
        "scope": "rf_xgboost_full_reference_search_space",
        "decision_record": "docs/P7C2_RF_XGBOOST_DECISION.md",
        "source_manifest": "configs/protocols/p7a/p7a_candidate_manifest.yaml",
    }:
        _error("final_manifest.protocol: unexpected final protocol identity.")
    p7a = load_manifest(root / protocol["source_manifest"])
    if payload.get("source_manifest_sha256") != manifest_hash(p7a):
        _error("final_manifest.source_manifest_sha256: must bind the P7A manifest.")
    if payload.get("decision_approval") != {
        "source": "user_task_instruction",
        "decision": "approve_full_reference_grids_no_reduced_grid_no_additional_pilot",
    }:
        _error("final_manifest.decision_approval: must record the approved decision provenance.")
    if payload.get("scientific_execution") != {
        "status": "not_authorized",
        "prerequisite": "p7c_3_through_p7c_7_readiness_gate",
    }:
        _error("final_manifest.scientific_execution: must preserve the P7C.7 execution gate.")
    models = payload.get("models")
    if not isinstance(models, list) or [item.get("id") for item in models if isinstance(item, dict)] != [
        "random_forest",
        "xgboost",
    ]:
        _error("final_manifest.models: must contain RF then XGBoost.")
    references = p7a["reference_search_spaces"]
    for item, model_id, implementation, count in (
        (models[0], "random_forest", "sklearn.ensemble.RandomForestClassifier", 30),
        (models[1], "xgboost", "xgboost.XGBClassifier", 108),
    ):
        if not isinstance(item, dict) or item.get("implementation") != implementation:
                _error(f"final_manifest.models.{model_id}: implementation mismatch.")
        if item.get("search_space") != {
            "selection": "full_reference_grid",
            "candidate_count": count,
            "parameters": references[model_id]["parameters"],
        }:
                _error(f"final_manifest.models.{model_id}: must exactly preserve the P7A reference grid.")
    workload = payload.get("workload")
    if workload != {
        "outer_partitions": 90,
        "inner_folds": 5,
        "random_forest": {
            "inner_candidate_evaluation_fits": 13500,
            "outer_selected_model_refits": 90,
            "total_estimator_fits": 13590,
        },
        "xgboost": {
            "inner_candidate_evaluation_fits": 48600,
            "outer_selected_model_refits": 90,
            "total_estimator_fits": 48690,
        },
        "combined_total_estimator_fits": 62280,
    }:
        _error("final_manifest.workload: workload must match the full reference grids.")
    lock = payload.get("lock")
    if not isinstance(lock, dict) or lock.get("algorithm") != "sha256-canonical-json" or lock.get(
        "manifest_sha256"
    ) != final_manifest_hash(payload):
        _error("final_manifest.lock: manifest hash mismatch.")
    return deepcopy(payload)


def load_rf_xgboost_final_manifest(
    path: str | Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise P7CInventoryError(f"cannot read final manifest: {exc}") from exc
    return validate_rf_xgboost_final_manifest(payload, repo_root=repo_root)


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
    for item in models:
        model_id = item["model_id"]
        if item["search_space_status"] == "locked":
            expected_count = LOCKED_FINAL_MODELS.get(model_id)
            if expected_count is None or item["candidate_count"] != expected_count:
                _error(f"{model_id}: invalid locked-model candidate count.")
            _relative_reference(
                item["final_manifest_reference"],
                root,
                f"{model_id}.final_manifest_reference",
                required=True,
            )
            _relative_reference(
                item["decision_record_reference"],
                root,
                f"{model_id}.decision_record_reference",
                required=True,
            )
            if item["blockers"]:
                _error(f"locked {model_id} must not retain unresolved blockers.")
        if (
            item["search_space_status"] != "locked"
            and item["compute_budget_status"] == "locked"
        ):
            _error(
                f"{item['model_id']}: unresolved search space cannot have a locked budget."
            )
    rf = models[1]
    xgb = models[2]
    if rf["search_space_status"] == xgb["search_space_status"] == "locked":
        if rf["final_manifest_reference"] != xgb["final_manifest_reference"]:
            _error("RF/XGBoost must share one final manifest.")
        if rf["decision_record_reference"] != xgb["decision_record_reference"]:
            _error("RF/XGBoost must share one decision record.")
        final_manifest = load_rf_xgboost_final_manifest(
            root / rf["final_manifest_reference"], repo_root=root
        )
        if final_manifest["models"][0]["search_space"]["candidate_count"] != rf["candidate_count"] or final_manifest[
            "models"
        ][1]["search_space"]["candidate_count"] != xgb["candidate_count"]:
            _error("RF/XGBoost inventory counts disagree with final manifest.")
    return deepcopy(payload)


def load_protocol_inventory(
    path: str | Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise P7CInventoryError(f"cannot read inventory: {exc}") from exc
    return validate_protocol_inventory(payload, repo_root=repo_root)
