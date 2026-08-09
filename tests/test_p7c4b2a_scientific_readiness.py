from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from creditrep.protocols.p7c4b2a import (
    P7C4B2AError,
    build_shard_plan,
    execution_guard,
    load_manifest,
    materialize_manifest,
    validate_manifest,
    validate_shard_plan,
)


MANIFEST = Path("configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml")


def payload():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def test_balanced_manifest_replays_exact_candidates_and_workload():
    first = load_manifest(MANIFEST)
    second = materialize_manifest(payload())
    assert first["models"] == second["models"]
    assert [item["candidate_count"] for item in first["models"]] == [24, 48, 48]
    assert [sum(candidate["selection_role"] == "mandatory" for candidate in item["candidates"]) for item in first["models"]] == [12, 18, 18]
    assert first["workload"]["total_inner_fits"] == 54000
    assert first["workload"]["total_outer_refits"] == 270
    assert first["workload"]["total_estimator_fits"] == 54270


@pytest.mark.parametrize("mutator,match", [
    (lambda value: value.__setitem__("compute_feasibility", "passed"), "compute feasibility"),
    (lambda value: value["scientific_approval"].__setitem__("approver", "other"), "approval authority"),
    (lambda value: value["lock"].__setitem__("manifest_sha256", "0" * 64), "digest mismatch"),
])
def test_manifest_fails_closed_on_contract_mutation(mutator, match):
    value = payload(); mutator(value)
    with pytest.raises(P7C4B2AError, match=match): validate_manifest(value)


def test_execution_guard_remains_closed_without_preflight_and_execution_approval():
    manifest = load_manifest(MANIFEST)
    report = execution_guard(manifest)
    assert report["authorized"] is False
    assert {"compute_preflight_pending", "canonical_compute_mode_pending", "execution_cost_approval_missing", "approved_execution_plan_missing"} <= set(report["reason_codes"])
    gpu = execution_guard(manifest, requested_mode="gpu_sequential")
    assert "gpu_not_authorized" in gpu["reason_codes"]


def test_static_two_vm_shards_are_deterministic_complete_and_non_overlapping():
    manifest = load_manifest(MANIFEST)
    first = build_shard_plan(manifest, 2)
    second = build_shard_plan(manifest, 2)
    assert first == second
    assert validate_shard_plan(first, manifest) == {"valid": True, "work_units": 90, "shards": 2}
    foreign = deepcopy(first); foreign["scientific_manifest_digest"] = "0" * 64
    with pytest.raises(P7C4B2AError, match="foreign scientific manifest digest"):
        validate_shard_plan(foreign, manifest)
    duplicate = deepcopy(first); duplicate["shards"][1]["work_units"].append(duplicate["shards"][0]["work_units"][0])
    with pytest.raises(P7C4B2AError, match="shard overlap"):
        validate_shard_plan(duplicate, manifest)
