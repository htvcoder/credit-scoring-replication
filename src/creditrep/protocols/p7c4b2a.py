"""Pure planning contracts for P7C.4B.2a; this module never trains models."""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Any

import yaml

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.registry import find_repo_root
from creditrep.protocols.p7a import load_manifest as load_p7a_manifest, manifest_hash


class P7C4B2AError(ValueError):
    pass


MODEL_SPECS = {"mlp_1": (1, 24, 12), "mlp_3": (3, 48, 18), "mlp_5": (5, 48, 18)}
REASON_CODES = {
    "compute_preflight_pending", "canonical_compute_mode_pending",
    "execution_cost_approval_missing", "approved_execution_plan_missing",
    "gpu_not_authorized", "multi_vm_not_authorized", "multi_vm_readiness_pending",
    "scientific_manifest_digest_mismatch", "execution_plan_digest_mismatch",
}


def _fail(message: str) -> None:
    raise P7C4B2AError(message)


def _candidate_population(reference: dict[str, Any], model_id: str, depth: int) -> list[dict[str, Any]]:
    shared = reference["shared"]
    widths = sorted(shared["hidden_units"])
    # combinations_with_replacement is ordered ascending; reverse each sequence so
    # the non-increasing architecture convention is explicit and canonical.
    units = [list(reversed(item)) for item in combinations_with_replacement(widths, depth)]
    values = []
    for hidden, dropout, l2, learning_rate in product(
        units, shared["dropout"], shared["l2"], reference[model_id]["learning_rate"]
    ):
        values.append({"hidden_units": hidden, "dropout": dropout, "l2": l2,
                       "learning_rate": learning_rate})
    return values


def _candidate_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (tuple(item["hidden_units"]), item["dropout"], item["l2"], item["learning_rate"])


def _mandatory(population: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministic coverage anchors, then canonical evenly-spaced fill."""
    indexes = [0, len(population) - 1, len(population) // 2]
    # These positions force coverage of endpoint/central regions under the stable
    # population ordering; remaining members are a deterministic evenly-spaced fill.
    for index in range(count):
        indexes.append(round(index * (len(population) - 1) / max(count - 1, 1)))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for index in indexes:
        item = population[index]
        if _candidate_key(item) not in seen:
            selected.append(item); seen.add(_candidate_key(item))
        if len(selected) == count:
            return selected
    for item in population:
        if _candidate_key(item) not in seen:
            selected.append(item); seen.add(_candidate_key(item))
        if len(selected) == count:
            return selected
    _fail("insufficient mandatory candidate population")


def _seeded_order(items: list[dict[str, Any]], seed: int, model_id: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: sha256_canonical({"root_seed": seed, "model_id": model_id, "tuple": _candidate_key(item)}))


def generate_candidates(reference: dict[str, Any], model_id: str, *, seed: int = 42) -> list[dict[str, Any]]:
    """Generate the approved balanced subset without data, metrics, or randomness."""
    depth, total, mandatory_count = MODEL_SPECS[model_id]
    population = _candidate_population(reference, model_id, depth)
    mandatory = _mandatory(population, mandatory_count)
    mandatory_keys = {_candidate_key(item) for item in mandatory}
    exploration = _seeded_order([item for item in population if _candidate_key(item) not in mandatory_keys], seed, model_id)[:total - mandatory_count]
    result = []
    for index, item in enumerate([*mandatory, *exploration]):
        # batch normalization is deliberately metadata, not a search dimension;
        # alternating values makes its runtime setting auditable without doubling
        # P7A's declared configuration count.
        result.append({"candidate_id": f"{model_id}-balanced-{index:03d}", **item,
                       "batch_normalization": bool(index % 2),
                       "selection_role": "mandatory" if index < mandatory_count else "seeded_stratified"})
    return result


def workload(candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_id, items in candidates.items():
        inner = len(items) * 90 * 5
        models[model_id] = {"inner_candidate_evaluation_fits": inner,
                            "outer_selected_model_refits": 90,
                            "total_estimator_fits": inner + 90}
    return {"outer_partitions": 90, "inner_folds": 5, "models": models,
            "total_inner_fits": sum(item["inner_candidate_evaluation_fits"] for item in models.values()),
            "total_outer_refits": sum(item["outer_selected_model_refits"] for item in models.values()),
            "total_estimator_fits": sum(item["total_estimator_fits"] for item in models.values())}


def _digest_payload(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value); result.pop("lock", None)
    # An approval must cite the final digest, so its citation is excluded from
    # the self-hash while the approver identity and decision remain protected.
    if isinstance(result.get("scientific_approval"), dict):
        result["scientific_approval"].pop("manifest_sha256", None)
    return result


def manifest_digest(value: dict[str, Any]) -> str:
    return sha256_canonical(_digest_payload(value))


def materialize_manifest(payload: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    p7a = load_p7a_manifest(root / payload["source_manifest"])
    result = deepcopy(payload)
    candidates = {model_id: generate_candidates(p7a["reference_search_spaces"]["mlp"], model_id) for model_id in MODEL_SPECS}
    result["models"] = [{"id": model_id, "hidden_depth": MODEL_SPECS[model_id][0],
                         "candidate_count": len(candidates[model_id]), "candidates": candidates[model_id]}
                        for model_id in MODEL_SPECS]
    result["workload"] = workload(candidates)
    return result


def validate_manifest(payload: Any, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        _fail("scientific manifest schema_version must be 1")
    if payload.get("scientific_scope") != "approved" or payload.get("canonical_execution", {}).get("status") != "not_authorized":
        _fail("scientific scope must be approved while canonical execution remains not_authorized")
    if payload.get("compute_feasibility") != "pending_preflight" or payload.get("canonical_compute_mode") != "pending_compute_preflight":
        _fail("compute feasibility and canonical mode must remain pending")
    if payload.get("source_manifest") != "configs/protocols/p7a/p7a_candidate_manifest.yaml":
        _fail("scientific manifest must bind P7A")
    p7a = load_p7a_manifest(root / payload["source_manifest"])
    if payload.get("source_manifest_sha256") != manifest_hash(p7a): _fail("P7A digest mismatch")
    materialized = materialize_manifest(payload, repo_root=root)
    models = materialized["models"]
    if [item["id"] for item in models] != list(MODEL_SPECS): _fail("unexpected MLP models")
    reference = p7a["reference_search_spaces"]["mlp"]
    for item in models:
        model_id = item["id"]; depth, count, mandatory_count = MODEL_SPECS[model_id]
        entries = item["candidates"]
        if len(entries) != count or len({_candidate_key(x) for x in entries}) != count:
            _fail(f"{model_id} candidate count or uniqueness mismatch")
        if sum(x["selection_role"] == "mandatory" for x in entries) != mandatory_count:
            _fail(f"{model_id} mandatory candidate count mismatch")
        if any(len(x["hidden_units"]) != depth or x["dropout"] not in reference["shared"]["dropout"] or x["l2"] not in reference["shared"]["l2"] or x["learning_rate"] not in reference[model_id]["learning_rate"] for x in entries):
            _fail(f"{model_id} candidate outside P7A reference grid")
    expected = {"total_inner_fits": 54000, "total_outer_refits": 270, "total_estimator_fits": 54270}
    if {key: materialized["workload"][key] for key in expected} != expected: _fail("balanced workload mismatch")
    approval = payload.get("scientific_approval", {})
    if approval.get("approver") != "Hoàng Trọng Vĩnh" or approval.get("reviewer") != {"name": "Trần Công Phú Khánh", "status": "pending"}:
        _fail("scientific approval authority mismatch")
    # The lock is over the resolved tuple list, not merely the generator policy.
    digest = manifest_digest(materialized)
    if payload.get("lock", {}).get("manifest_sha256") != digest or approval.get("manifest_sha256") != digest: _fail("scientific manifest digest mismatch")
    return materialized


def load_manifest(path: str | Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    try: payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc: raise P7C4B2AError(f"cannot read scientific manifest: {exc}") from exc
    return validate_manifest(payload, repo_root=repo_root)


def execution_guard(manifest: dict[str, Any], execution_plan: dict[str, Any] | None = None, *, requested_mode: str = "cpu_parallel_2") -> dict[str, Any]:
    """Fail closed; it is deliberately a planning-time guard, not an executor."""
    codes: list[str] = []
    if manifest.get("compute_feasibility") != "passed": codes.append("compute_preflight_pending")
    if manifest.get("canonical_compute_mode") == "pending_compute_preflight": codes.append("canonical_compute_mode_pending")
    approval = manifest.get("execution_cost_approval")
    if not approval: codes.append("execution_cost_approval_missing")
    if not execution_plan: codes.append("approved_execution_plan_missing")
    if requested_mode.startswith("gpu"): codes.append("gpu_not_authorized")
    if requested_mode == "multi_vm_cpu_parallel_2": codes.extend(["multi_vm_not_authorized", "multi_vm_readiness_pending"])
    return {"authorized": not codes, "reason_codes": sorted(set(codes))}


def build_shard_plan(manifest: dict[str, Any], shard_count: int) -> dict[str, Any]:
    """Static deterministic shard planning at dataset × outer-split granularity."""
    if shard_count < 1: _fail("shard count must be positive")
    digest = manifest["lock"]["manifest_sha256"]
    repeats = {"AC": 10, "GC": 10, "TH02": 10, "HMEQ": 5, "TC": 5, "GMC": 5}
    units = [{"dataset_id": dataset, "outer_repeat": repeat, "outer_fold": fold,
              "work_unit_id": sha256_canonical({"scientific_manifest_digest": digest, "dataset_id": dataset, "outer_repeat": repeat, "outer_fold": fold})}
             for dataset, count in repeats.items() for repeat in range(count) for fold in range(2)]
    shards = [{"shard_id": f"shard-{index + 1:02d}-of-{shard_count:02d}", "work_units": []} for index in range(shard_count)]
    for unit in sorted(units, key=lambda x: x["work_unit_id"]):
        shard = shards[int(unit["work_unit_id"], 16) % shard_count]
        shard["work_units"].append({**unit, "shard_id": shard["shard_id"]})
    for shard in shards:
        shard["shard_manifest_digest"] = sha256_canonical({"scientific_manifest_digest": digest, "shard_id": shard["shard_id"], "work_units": shard["work_units"]})
    result = {"schema_version": 1, "scientific_manifest_digest": digest,
              "partition_key": "dataset_outer_repeat_outer_fold",
              "logical_fit_identity_contract": ["scientific_manifest_digest", "dataset_id", "dataset_fingerprint", "model_id", "outer_repeat", "outer_fold", "candidate_id", "inner_fold", "seed_identity", "shard_id"],
              "provenance": {"planner": "p7c4b2a_static_sharding_v1", "assignment": "sha256_work_unit_modulo_shard_count"},
              "shards": shards}
    result["partition_plan_digest"] = sha256_canonical(result)
    return result


def validate_shard_plan(plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    if plan.get("scientific_manifest_digest") != manifest["lock"]["manifest_sha256"]: _fail("foreign scientific manifest digest")
    units = [item["work_unit_id"] for shard in plan.get("shards", []) for item in shard.get("work_units", [])]
    expected = [item["work_unit_id"] for shard in build_shard_plan(manifest, len(plan.get("shards", []))).get("shards", []) for item in shard["work_units"]]
    if len(units) != len(set(units)): _fail("shard overlap detected")
    if set(units) != set(expected): _fail("shard completeness mismatch")
    for shard in plan["shards"]:
        expected_digest = sha256_canonical({"scientific_manifest_digest": manifest["lock"]["manifest_sha256"], "shard_id": shard["shard_id"], "work_units": shard["work_units"]})
        if shard.get("shard_manifest_digest") != expected_digest: _fail("shard manifest digest mismatch")
    if plan.get("partition_plan_digest") != sha256_canonical({k: v for k, v in plan.items() if k != "partition_plan_digest"}): _fail("partition plan digest mismatch")
    return {"valid": True, "work_units": len(units), "shards": len(plan["shards"])}
