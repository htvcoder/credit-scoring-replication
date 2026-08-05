"""Read-only validation for the immutable P7C.2 RF/XGBoost pilot plan."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any
import yaml
from creditrep.config.loader import sha256_canonical
from creditrep.protocols.p7a import load_manifest, manifest_hash


class P7C2PlanError(ValueError):
    pass


def plan_digest(plan: dict[str, Any]) -> str:
    value = deepcopy(plan)
    value.pop("lock", None)
    return sha256_canonical(value)


def validate_pilot_plan(plan: Any, *, repo_root: Path) -> dict[str, Any]:
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != 1
        or plan.get("checkpoint_id") != "P7C.2.1"
    ):
        raise P7C2PlanError("invalid schema/checkpoint")
    if plan.get("status") != "planned" or plan.get("execution_status") != "not_run":
        raise P7C2PlanError("plan must remain planned/not_run")
    if (
        plan.get("scientific_boundary")
        != "no_predictive_ranking_no_selection_no_outer_refit_non_publishable"
    ):
        raise P7C2PlanError("scientific boundary mismatch")
    if (
        Path(str(plan.get("artifact_root", ""))).is_absolute()
        or PureWindowsPath(str(plan.get("artifact_root", ""))).is_absolute()
        or plan.get("artifact_root") != "artifacts/p7c2-rf-xgboost-feasibility"
    ):
        raise P7C2PlanError("invalid artifact root")
    p7a = load_manifest(repo_root / "configs/protocols/p7a/p7a_candidate_manifest.yaml")
    if plan.get(
        "source_manifest"
    ) != "configs/protocols/p7a/p7a_candidate_manifest.yaml" or plan.get(
        "source_manifest_sha256"
    ) != manifest_hash(p7a):
        raise P7C2PlanError("reference manifest mismatch")
    if (
        plan.get("datasets") != ["AC", "GMC"]
        or plan.get("outer_partition", {}).get("repeat_index") != 0
        or plan.get("outer_partition", {}).get("fold_index") != 0
        or plan.get("inner_folds") != 5
    ):
        raise P7C2PlanError("dataset/partition/inner-fold contract mismatch")
    thread = plan.get("threading", {})
    if thread != {
        "fits_parallelism": 1,
        "estimator_threads": 1,
        "allow_n_jobs_minus_one": False,
        "gpu_enabled": False,
    }:
        raise P7C2PlanError("nested parallelism/GPU policy mismatch")
    models = plan.get("models")
    if not isinstance(models, list) or [x.get("model_id") for x in models] != [
        "random_forest",
        "xgboost",
    ]:
        raise P7C2PlanError("model IDs must be RF then XGBoost")
    grids = p7a["reference_search_spaces"]
    counts = {}
    for model in models:
        candidates = model.get("candidates", [])
        ids = [x.get("id") for x in candidates]
        if len(candidates) != 3 or len(set(ids)) != 3:
            raise P7C2PlanError("candidate IDs must be three and unique")
        source = grids[model["model_id"]]["parameters"]
        for candidate in candidates:
            params = candidate.get("parameters", {})
            if model["model_id"] == "random_forest":
                ok = (
                    set(params) == {"n_estimators", "max_features_multiplier_of_sqrt_m"}
                    and params["n_estimators"] in source["n_estimators"]
                    and params["max_features_multiplier_of_sqrt_m"]
                    in source["max_features_multiplier_of_sqrt_m"]
                )
            else:
                ok = set(params) == set(source) and all(
                    params[k] in source[k] for k in source
                )
            if not ok:
                raise P7C2PlanError(
                    f"candidate outside reference grid: {candidate.get('id')}"
                )
        counts[model["model_id"]] = (
            len(plan["datasets"]) * len(candidates) * plan["inner_folds"]
        )
    if plan.get("expected_fits") != {
        "random_forest": counts["random_forest"],
        "xgboost": counts["xgboost"],
        "total": sum(counts.values()),
    }:
        raise P7C2PlanError("expected fit count mismatch")
    if plan.get("lock", {}).get("algorithm") != "sha256-canonical-json" or plan[
        "lock"
    ].get("plan_sha256") != plan_digest(plan):
        raise P7C2PlanError("plan digest mismatch")
    return deepcopy(plan)


def load_pilot_plan(path: str | Path, *, repo_root: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    except (OSError, yaml.YAMLError) as exc:
        raise P7C2PlanError(f"cannot read plan: {exc}") from exc
    return validate_pilot_plan(payload, repo_root=repo_root)
