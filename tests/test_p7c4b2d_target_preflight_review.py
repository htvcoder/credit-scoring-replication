from __future__ import annotations

from concurrent.futures import Future
from datetime import UTC, datetime, timedelta
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest
import psutil

from creditrep.experiments import p7c4b2d_cli as cli
from creditrep.experiments import p7c4b2c_cli as preflight_cli
from creditrep.experiments import p7c4b2c_preflight as runner
from creditrep.experiments.p7c4b2c_cli import main as preflight_cli_main
from creditrep.experiments.p7c4b2d_cli import EXIT_REVIEW_BLOCKED, main as cli_main
from creditrep.locked_runtime_inputs import (
    LockedRuntimeInputError,
    OUTER_PROJECTION_RUNTIME_DATASETS,
    load_locked_runtime_inputs,
)
from creditrep.preprocessing import load_protocol_a_config
from creditrep.preprocessing.exceptions import PreprocessingError
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2c import P7C4B2CError, TIMING_FIELDS, build_plan
from creditrep.protocols.p7c4b2c import canonical_digest
from creditrep.protocols.p7c4b2d import (
    DISK_POLICY,
    MINIMUM_FREE_DISK_BYTES,
    P7C4B2DError,
    PROJECTION_AGGREGATE_RSS_LIMIT_BYTES,
    TARGET_CANARY_APPROVAL,
    TARGET_PROJECTION_PREFLIGHT_APPROVAL,
    TARGET_PROJECTION_PREFLIGHT_STAGE,
    _create_effective_authorization,
    _static_authorization_cost_upper_bound,
    collect_target_environment,
    create_effective_authorization,
    dependency_lock_fingerprint,
    decision_package,
    dataset_binding_digest,
    environment_digest,
    merge_operator_metadata,
    normalize_target_output,
    projection_preflight_resource_policy,
    render_authorization_proposal,
    select_canary,
    select_projection_preflight,
    validate_authorization_proposal,
    validate_effective_authorization,
    validate_target_environment,
)


def _plan():
    return build_plan(
        load_manifest("configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml")
    )


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['setuptools>=69']\n", encoding="utf-8"
    )
    (tmp_path / "requirements.txt").write_text("numpy==2\n", encoding="utf-8")
    (tmp_path / "requirements-dev.txt").write_text(
        "-r requirements.txt\n", encoding="utf-8"
    )
    dataset_names = ("ac", "gc", "th02", "hmeq", "tc", "gmc")
    for dataset in dataset_names:
        payload = f"{dataset} fixture\n".encode()
        folder = tmp_path / "data" / "raw" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{dataset}.csv").write_bytes(payload)
    registry = """datasets:
  ac:
    id: ac
    active_file: data/raw/ac/ac.csv
    reader: {type: csv}
    target: {column: target, mapping_to_binary: {'0': 0, '1': 1}}
  gc:
    id: gc
    active_file: data/raw/gc/gc.csv
    reader: {type: csv}
    target: {column: target, mapping_to_binary: {'0': 0, '1': 1}}
  th02:
    id: th02
    active_file: data/raw/th02/th02.csv
    reader: {type: csv}
    target: {column: target, mapping_to_binary: {'0': 0, '1': 1}}
  hmeq:
    id: hmeq
    active_file: data/raw/hmeq/hmeq.csv
    reader: {type: csv}
    target: {column: target, mapping_to_binary: {'0': 0, '1': 1}}
  tc:
    id: tc
    active_file: data/raw/tc/tc.csv
    reader: {type: csv}
    target: {column: target, mapping_to_binary: {'0': 0, '1': 1}}
  gmc:
    id: gmc
    active_file: data/raw/gmc/gmc.csv
    reader: {type: csv}
    target: {column: target, mapping_to_binary: {'0': 0, '1': 1}}
"""
    (tmp_path / "data" / "datasets.yaml").write_text(registry, encoding="utf-8")
    checks = ['"Path","Algorithm","Hash"']
    for dataset in dataset_names:
        payload = (tmp_path / "data" / "raw" / dataset / f"{dataset}.csv").read_bytes()
        checks.append(
            f'"data/raw/{dataset}/{dataset}.csv","SHA256","{hashlib.sha256(payload).hexdigest().upper()}"'
        )
    (tmp_path / "data" / "checksums-sha256.csv").write_text(
        "\n".join(checks) + "\n", encoding="utf-8"
    )
    (tmp_path / "artifacts").mkdir()
    source_root = Path(__file__).resolve().parents[1]
    for relative in (
        Path("configs/protocols/protocol_a.yaml"),
        Path("configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml"),
        Path("configs/protocols/p7a/p7a_candidate_manifest.yaml"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
    return tmp_path


def _environment(plan, root: Path, monkeypatch, *, mode="cpu_parallel_1"):
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.current_git_head", lambda _: "a" * 40
    )
    hashes = {
        "AC": hashlib.sha256((root / "data/raw/ac/ac.csv").read_bytes())
        .hexdigest()
        .upper(),
        "GMC": hashlib.sha256((root / "data/raw/gmc/gmc.csv").read_bytes())
        .hexdigest()
        .upper(),
    }
    value = {
        "schema_version": 2,
        "artifact_type": "target_environment",
        "checkpoint": "P7C.4B.2d",
        "provider": "fixture",
        "region": "fixture",
        "instance_id": "fixture",
        "os": "fixture",
        "python_version": "3.11",
        "cpu_model": "fixture",
        "vcpu_count": 2,
        "ram_bytes": 16 * 1024**3,
        "gpu_model": "none",
        "gpu_count": 0,
        "gpu_vram_bytes": 0,
        "disk_type": "fixture",
        "free_disk_bytes": MINIMUM_FREE_DISK_BYTES + 1,
        "network_topology": "single_vm",
        "disk_policy": DISK_POLICY,
        "worker_count": 1 if mode == "cpu_parallel_1" else 2,
        "execution_mode": mode,
        "vm_count": 1,
        "git_commit": "a" * 40,
        "expected_git_commit": "a" * 40,
        "plan_digest": plan["plan_digest"],
        "environment_lock_hash": dependency_lock_fingerprint(root)["sha256"],
        "dataset_hashes": hashes,
        "locked_runtime_inputs_digest": load_locked_runtime_inputs(root).digest,
        "output_directory": "artifacts/p7c4b2d-fixture",
        "hourly_price": "2.5",
        "currency": "USD",
        "price_source": "fixture",
        "price_observed_at": "2026-08-10T00:00:00Z",
        "maximum_runtime_hours": 10.0,
        "maximum_monetary_budget": "25.0",
        "projection_preflight_resource_policy": projection_preflight_resource_policy(
            mode
        ),
        "evidence_observed_at": "2026-08-10T00:00:00Z",
        "process_spawn_probe": _pass_probe(),
    }
    value["environment_digest"] = environment_digest(value)
    return value


def _as_projection_environment(value, plan, root):
    value["maximum_runtime_hours"] = 12
    value["maximum_monetary_budget"] = "5.0"
    value["hourly_price"] = "0.26"
    value["ram_bytes"] = PROJECTION_AGGREGATE_RSS_LIMIT_BYTES + 1
    runtime_inputs = load_locked_runtime_inputs(
        root, dataset_ids=OUTER_PROJECTION_RUNTIME_DATASETS
    )
    value.update(
        {
            "schema_version": 3,
            "execution_stage": TARGET_PROJECTION_PREFLIGHT_STAGE,
            "dataset_ids": list(OUTER_PROJECTION_RUNTIME_DATASETS),
            "dataset_hashes": runtime_inputs.source_hashes,
            "locked_runtime_inputs_digest": runtime_inputs.digest,
        }
    )
    value["dataset_binding_digest"] = dataset_binding_digest(
        dataset_ids=OUTER_PROJECTION_RUNTIME_DATASETS,
        dataset_hashes=value["dataset_hashes"],
        locked_runtime_inputs_digest=value["locked_runtime_inputs_digest"],
        plan_digest=plan["plan_digest"],
    )
    value["environment_digest"] = environment_digest(value)
    return value


def _pass_probe():
    return {"probe": "fixture", "status": "pass", "timeout_seconds": 1}


def _spawn_sleeping_descendant(pid_queue):
    descendant = get_context("spawn").Process(target=time.sleep, args=(30,))
    descendant.start()
    pid_queue.put((os.getpid(), descendant.pid))
    time.sleep(30)


def _operator_metadata():
    return {
        "provider": "fixture-cloud",
        "region": "fixture-region",
        "instance_id": "fixture-instance",
        "disk_type": "ssd",
        "network_topology": "single_vm",
        "vm_count": 1,
        "hourly_price": "2.5",
        "currency": "USD",
        "price_source": "fixture-invoice",
        "price_observed_at": "2026-08-10T00:00:00Z",
        "maximum_runtime_hours": 10.0,
        "maximum_monetary_budget": "25.0",
    }


def _proposal_and_authorization(plan, root, monkeypatch):
    environment = _environment(plan, root, monkeypatch)
    proposal = render_authorization_proposal(
        plan, environment, execution_stage="target_canary", expiry=None
    )
    authorization = _create_effective_authorization(
        proposal,
        environment,
        plan,
        operator_identity="fixture-operator",
        operator_approval=TARGET_CANARY_APPROVAL,
        expires_at="2026-08-10T01:00:00Z",
        repo_root=root,
        spawn_probe=_pass_probe,
        now_provider=lambda: datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
    )
    return environment, proposal, authorization


def test_stage_zero_verifies_canonical_sources_and_rejects_placeholders(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    assert validate_target_environment(
        environment, plan, repo_root=root, spawn_probe=_pass_probe
    )["valid"]
    environment["dataset_hashes"] = {"AC": "fixture", "GMC": "fixture"}
    environment["environment_digest"] = environment_digest(environment)
    report = validate_target_environment(
        environment, plan, repo_root=root, spawn_probe=_pass_probe
    )
    assert "dataset_input_hash_mismatch" in report["reason_codes"]


@pytest.mark.parametrize("mode", ["cpu_parallel_1", "cpu_parallel_2"])
def test_projection_preflight_selection_is_deterministic_and_minimal(mode):
    plan = _plan()
    full = select_projection_preflight(plan, mode)

    assert full["execution_stage"] == TARGET_PROJECTION_PREFLIGHT_STAGE
    assert full["task_count"] == 162
    assert sum(task["classification"] == "warmup" for task in full["tasks"]) == 54
    assert sum(task["classification"] == "measured" for task in full["tasks"]) == 108
    assert full["scientific_projection_eligible"] is False
    assert full["canonical_scientific_execution_authorized"] is False


@pytest.mark.parametrize(
    "stage,expected_count",
    [
        (TARGET_PROJECTION_PREFLIGHT_STAGE, 162),
    ],
)
def test_projection_preflight_proposal_is_typed_and_fail_closed(
    tmp_path, monkeypatch, stage, expected_count
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch, mode="cpu_parallel_2")
    _as_projection_environment(environment, plan, root)
    proposal = render_authorization_proposal(
        plan,
        environment,
        execution_stage=stage,
        expiry=None,
        repo_root=root,
        spawn_probe=_pass_probe,
    )

    assert proposal["maximum_task_count"] == expected_count
    assert len(proposal["task_ids"]) == expected_count
    assert validate_authorization_proposal(proposal, plan, environment)["valid"]
    proposal["task_ids"] = proposal["task_ids"][:-1]
    _redigest_proposal(proposal)
    assert (
        "authorization_proposal_task_scope_mismatch"
        in validate_authorization_proposal(proposal, plan, environment)["reason_codes"]
    )


def test_projection_preflight_requires_distinct_operator_approval(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    _as_projection_environment(environment, plan, root)
    proposal = render_authorization_proposal(
        plan,
        environment,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        expiry=None,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    with pytest.raises(P7C4B2DError, match="operator_approval_missing"):
        _create_effective_authorization(
            proposal,
            environment,
            plan,
            operator_identity="fixture-operator",
            operator_approval=TARGET_CANARY_APPROVAL,
            expires_at="2026-08-10T01:00:00Z",
            repo_root=root,
            spawn_probe=_pass_probe,
            now_provider=lambda: datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
        )
    authorization = _create_effective_authorization(
        proposal,
        environment,
        plan,
        operator_identity="fixture-operator",
        operator_approval=TARGET_PROJECTION_PREFLIGHT_APPROVAL,
        expires_at="2026-08-10T01:00:00Z",
        repo_root=root,
        spawn_probe=_pass_probe,
        now_provider=lambda: datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
    )
    assert authorization["execution_stage"] == TARGET_PROJECTION_PREFLIGHT_STAGE
    assert validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=datetime.fromisoformat("2026-08-10T00:30:00+00:00"),
        repo_root=root,
        spawn_probe=_pass_probe,
    )["valid"]


def test_projection_resource_policy_is_exact_and_per_run(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    first = _as_projection_environment(
        _environment(plan, root, monkeypatch, mode="cpu_parallel_1"), plan, root
    )
    second = _as_projection_environment(
        _environment(plan, root, monkeypatch, mode="cpu_parallel_2"), plan, root
    )
    first["output_directory"] = str(root / "artifacts" / "outer-p1")
    second["output_directory"] = str(root / "artifacts" / "outer-p2")
    first["environment_digest"] = environment_digest(first)
    second["environment_digest"] = environment_digest(second)
    proposals = [
        render_authorization_proposal(
            plan,
            value,
            execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
            expiry=None,
            repo_root=root,
            spawn_probe=_pass_probe,
        )
        for value in (first, second)
    ]
    assert [
        item["resource_policy"]["maximum_in_flight_tasks"] for item in proposals
    ] == [1, 2]
    assert all(item["maximum_runtime_hours"] == 12 for item in proposals)
    assert all(item["maximum_monetary_budget"] == "5.0" for item in proposals)
    assert sum(item["maximum_runtime_hours"] for item in proposals) == 24
    assert sum(float(item["maximum_monetary_budget"]) for item in proposals) == 10
    assert (
        proposals[0]["target_environment_digest"]
        != proposals[1]["target_environment_digest"]
    )


@pytest.mark.parametrize("missing_dataset", OUTER_PROJECTION_RUNTIME_DATASETS)
def test_outer_environment_rejects_each_missing_dataset(
    tmp_path, monkeypatch, missing_dataset
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _as_projection_environment(
        _environment(plan, root, monkeypatch), plan, root
    )
    environment["dataset_ids"].remove(missing_dataset)
    environment["dataset_hashes"].pop(missing_dataset)
    environment["dataset_binding_digest"] = dataset_binding_digest(
        dataset_ids=environment["dataset_ids"],
        dataset_hashes=environment["dataset_hashes"],
        locked_runtime_inputs_digest=environment["locked_runtime_inputs_digest"],
        plan_digest=environment["plan_digest"],
    )
    environment["environment_digest"] = environment_digest(environment)
    report = validate_target_environment(
        environment,
        plan,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert not report["valid"]
    assert "outer_dataset_inventory_mismatch" in report["reason_codes"]


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "lowercase", "wrong_hash"])
def test_outer_environment_inventory_and_hash_mutations_fail_closed(
    tmp_path, monkeypatch, mutation
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _as_projection_environment(
        _environment(plan, root, monkeypatch), plan, root
    )
    if mutation == "extra":
        environment["dataset_ids"].append("EXTRA")
        environment["dataset_hashes"]["EXTRA"] = "A" * 64
    elif mutation == "duplicate":
        environment["dataset_ids"].append("GC")
    elif mutation == "lowercase":
        environment["dataset_ids"][1] = "gc"
        environment["dataset_hashes"]["gc"] = environment["dataset_hashes"].pop("GC")
    else:
        environment["dataset_hashes"]["GC"] = "0" * 64
    environment["dataset_binding_digest"] = dataset_binding_digest(
        dataset_ids=environment["dataset_ids"],
        dataset_hashes=environment["dataset_hashes"],
        locked_runtime_inputs_digest=environment["locked_runtime_inputs_digest"],
        plan_digest=environment["plan_digest"],
    )
    environment["environment_digest"] = environment_digest(environment)
    report = validate_target_environment(
        environment,
        plan,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert not report["valid"]
    assert set(report["reason_codes"]) & {
        "outer_dataset_inventory_mismatch",
        "outer_dataset_hash_mismatch",
    }


def test_projection_dataset_hash_inventory_survives_json_key_reordering(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _as_projection_environment(
        _environment(plan, root, monkeypatch), plan, root
    )

    # This is the CLI persistence boundary: its JSON output uses sort_keys=True.
    persisted = json.loads(json.dumps(environment, sort_keys=True))
    assert list(persisted["dataset_hashes"]) != list(OUTER_PROJECTION_RUNTIME_DATASETS)
    report = validate_target_environment(
        persisted,
        plan,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert report["valid"]

    reordered = dict(reversed(list(environment["dataset_hashes"].items())))
    environment["dataset_hashes"] = reordered
    environment["environment_digest"] = environment_digest(environment)
    assert validate_target_environment(
        environment,
        plan,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        repo_root=root,
        spawn_probe=_pass_probe,
    )["valid"]


@pytest.mark.parametrize(
    "mutation,expected_code",
    [
        ("missing", "outer_dataset_inventory_mismatch"),
        ("extra", "outer_dataset_inventory_mismatch"),
        ("wrong_hash", "outer_dataset_hash_mismatch"),
        ("malformed_hash", "outer_dataset_inventory_mismatch"),
        ("wrong_locked_digest", "locked_runtime_input_mismatch"),
        ("reordered_ids", "outer_dataset_inventory_mismatch"),
    ],
)
def test_projection_dataset_inventory_contract_remains_strict(
    tmp_path, monkeypatch, mutation, expected_code
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _as_projection_environment(
        _environment(plan, root, monkeypatch), plan, root
    )
    if mutation == "missing":
        environment["dataset_hashes"].pop("AC")
    elif mutation == "extra":
        environment["dataset_hashes"]["EXTRA"] = "A" * 64
    elif mutation == "wrong_hash":
        environment["dataset_hashes"]["AC"] = "0" * 64
    elif mutation == "malformed_hash":
        environment["dataset_hashes"]["AC"] = "not-a-sha256"
    elif mutation == "wrong_locked_digest":
        environment["locked_runtime_inputs_digest"] = "0" * 64
    else:
        environment["dataset_ids"] = list(reversed(environment["dataset_ids"]))
    environment["dataset_binding_digest"] = dataset_binding_digest(
        dataset_ids=environment["dataset_ids"],
        dataset_hashes=environment["dataset_hashes"],
        locked_runtime_inputs_digest=environment["locked_runtime_inputs_digest"],
        plan_digest=environment["plan_digest"],
    )
    environment["environment_digest"] = environment_digest(environment)
    report = validate_target_environment(
        environment,
        plan,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert expected_code in report["reason_codes"]


def test_outer_partial_historical_binding_fails_before_output_or_fit(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _as_projection_environment(
        _environment(plan, root, monkeypatch), plan, root
    )
    output = root / "artifacts" / "outer-regression"
    environment["output_directory"] = str(output)
    environment["environment_digest"] = environment_digest(environment)
    proposal = render_authorization_proposal(
        plan,
        environment,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        expiry=None,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    now = datetime.now(UTC)
    authorization = _create_effective_authorization(
        proposal,
        environment,
        plan,
        operator_identity="fixture-operator",
        operator_approval=TARGET_PROJECTION_PREFLIGHT_APPROVAL,
        expires_at=(now + timedelta(hours=1)).isoformat(),
        repo_root=root,
        spawn_probe=_pass_probe,
        now_provider=lambda: now,
    )
    partial_ids = ["AC", "GMC"]
    partial_hashes = {key: environment["dataset_hashes"][key] for key in partial_ids}
    wrong_binding = dataset_binding_digest(
        dataset_ids=partial_ids,
        dataset_hashes=partial_hashes,
        locked_runtime_inputs_digest=environment["locked_runtime_inputs_digest"],
        plan_digest=plan["plan_digest"],
    )
    for artifact in (environment, proposal, authorization):
        artifact["dataset_ids"] = partial_ids
        artifact["dataset_hashes"] = partial_hashes
        artifact["dataset_binding_digest"] = wrong_binding
    environment["environment_digest"] = environment_digest(environment)
    proposal["target_environment_digest"] = environment["environment_digest"]
    proposal["proposal_digest"] = canonical_digest(proposal, "proposal_digest")
    authorization["target_environment_digest"] = environment["environment_digest"]
    authorization["proposal_digest"] = proposal["proposal_digest"]
    authorization["authorization_digest"] = canonical_digest(
        authorization, "authorization_digest"
    )
    monkeypatch.setattr(
        runner,
        "canonical_outer_refit",
        lambda *_args, **_kwargs: pytest.fail("estimator dispatch must not occur"),
    )
    with pytest.raises(P7C4B2CError, match="outer_dataset_inventory_mismatch"):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert not output.exists()


def test_outer_recomputed_cross_binding_and_digest_mutations_are_rejected(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _as_projection_environment(
        _environment(plan, root, monkeypatch), plan, root
    )
    proposal = render_authorization_proposal(
        plan,
        environment,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        expiry=None,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    now = datetime.now(UTC)
    authorization = _create_effective_authorization(
        proposal,
        environment,
        plan,
        operator_identity="fixture-operator",
        operator_approval=TARGET_PROJECTION_PREFLIGHT_APPROVAL,
        expires_at=(now + timedelta(hours=1)).isoformat(),
        repo_root=root,
        spawn_probe=_pass_probe,
        now_provider=lambda: now,
    )
    authorization["dataset_hashes"]["GC"] = "0" * 64
    authorization["dataset_binding_digest"] = dataset_binding_digest(
        dataset_ids=authorization["dataset_ids"],
        dataset_hashes=authorization["dataset_hashes"],
        locked_runtime_inputs_digest=authorization["locked_runtime_inputs_digest"],
        plan_digest=authorization["plan_digest"],
    )
    authorization["authorization_digest"] = canonical_digest(
        authorization, "authorization_digest"
    )
    auth_report = validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=now + timedelta(minutes=1),
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert "outer_dataset_binding_mismatch" in auth_report["reason_codes"]
    proposal["dataset_hashes"]["GC"] = "0" * 64
    proposal["dataset_binding_digest"] = dataset_binding_digest(
        dataset_ids=proposal["dataset_ids"],
        dataset_hashes=proposal["dataset_hashes"],
        locked_runtime_inputs_digest=proposal["locked_runtime_inputs_digest"],
        plan_digest=proposal["plan_digest"],
    )
    proposal["proposal_digest"] = canonical_digest(proposal, "proposal_digest")
    assert (
        "outer_dataset_binding_mismatch"
        in validate_authorization_proposal(proposal, plan, environment)["reason_codes"]
    )

    environment["locked_runtime_inputs_digest"] = "0" * 64
    environment["dataset_binding_digest"] = dataset_binding_digest(
        dataset_ids=environment["dataset_ids"],
        dataset_hashes=environment["dataset_hashes"],
        locked_runtime_inputs_digest=environment["locked_runtime_inputs_digest"],
        plan_digest=environment["plan_digest"],
    )
    environment["environment_digest"] = environment_digest(environment)
    report = validate_target_environment(
        environment,
        plan,
        execution_stage=TARGET_PROJECTION_PREFLIGHT_STAGE,
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert "locked_runtime_input_mismatch" in report["reason_codes"]


def test_outer_locked_inputs_reject_source_symlink_escape(tmp_path):
    if os.name == "nt":
        pytest.skip("source symlink boundary requires a symlink-capable test host")
    (tmp_path / "repo").mkdir()
    root = _repo(tmp_path / "repo")
    outside = tmp_path / "outside-gc.csv"
    outside.write_bytes((root / "data/raw/gc/gc.csv").read_bytes())
    source = root / "data/raw/gc/gc.csv"
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(LockedRuntimeInputError, match="source_symlink_or_path_invalid"):
        load_locked_runtime_inputs(root, dataset_ids=OUTER_PROJECTION_RUNTIME_DATASETS)


def test_runtime_and_monetary_guards_are_independent():
    provenance = {
        "maximum_runtime_hours": 12,
        "maximum_monetary_budget": "5.0",
        "hourly_price": "0.26",
        "vm_count": 1,
    }
    assert (
        runner._elapsed_budget_violation(provenance, 43_200)
        == "runtime_budget_exceeded"
    )
    expensive = {**provenance, "hourly_price": "0.5"}
    assert (
        runner._elapsed_budget_violation(expensive, 36_000)
        == "monetary_budget_exceeded"
    )
    assert runner._elapsed_budget_violation(provenance, 1.0) is None


def test_killable_child_is_terminated_and_reaped():
    process = get_context("spawn").Process(target=time.sleep, args=(30,))
    process.start()
    runner._terminate_and_reap([process])
    assert process.is_alive() is False
    assert process.exitcode is not None


def test_killable_process_tree_terminates_and_reaps_descendant():
    context = get_context("spawn")
    pid_queue = context.Queue()
    process = context.Process(target=_spawn_sleeping_descendant, args=(pid_queue,))
    process.start()
    root_pid, descendant_pid = pid_queue.get(timeout=10)
    assert root_pid == process.pid
    assert psutil.pid_exists(descendant_pid)
    runner._terminate_and_reap([process])
    assert process.is_alive() is False
    assert not psutil.pid_exists(root_pid)
    assert not psutil.pid_exists(descendant_pid)
    pid_queue.close()


def test_supervisor_timeout_caps_inflight_and_stops_dispatch(tmp_path, monkeypatch):
    state = {"started": 0, "active": 0, "maximum": 0, "clock": 0.0}
    persisted_reasons = []

    class FakeQueue:
        pass

    class FakeProcess:
        pid = None
        exitcode = None

        def __init__(self, **_kwargs):
            self.alive = False

        def start(self):
            self.alive = True
            state["started"] += 1
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])

        def is_alive(self):
            return self.alive

        def terminate(self):
            if self.alive:
                state["active"] -= 1
            self.alive = False
            self.exitcode = -15

        kill = terminate

        def join(self, timeout=None):
            del timeout

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, **kwargs):
            return FakeProcess(**kwargs)

    def clock():
        state["clock"] += 601.0
        return state["clock"]

    monkeypatch.setattr(runner, "_persist_runtime_state", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner,
        "_persist_supervisor_failure",
        lambda _run, _manifest, _task, reason: persisted_reasons.append(reason),
    )
    policy = projection_preflight_resource_policy("cpu_parallel_2")
    manifest = {
        "run_id": "fixture-run",
        "worker_count": 2,
        "execution_class": "target_preflight",
        "plan_digest": "a" * 64,
        "authorization_provenance": {"git_commit": "b" * 40},
    }
    runtime_state = {
        "failed_task_count": 0,
        "accumulated_elapsed_seconds": 0.0,
        "last_accounted_at": "2026-08-13T00:00:00Z",
    }
    with pytest.raises(P7C4B2CError, match="task_timeout_exceeded"):
        runner._run_supervised_projection_phase(
            [{"sample_id": str(index)} for index in range(3)],
            manifest=manifest,
            environment={},
            run_dir=tmp_path,
            repo_root=tmp_path,
            provenance={
                "execution_stage": TARGET_PROJECTION_PREFLIGHT_STAGE,
                "resource_policy": policy,
            },
            runtime_state=runtime_state,
            target_task_gate=lambda: None,
            authorization_deadline=lambda: 1_000_000.0,
            monotonic_clock=clock,
            rss_sampler=lambda _pid: 1,
            context_provider=lambda _method: FakeContext(),
            poll_interval_seconds=0.0,
        )
    assert state["started"] == state["maximum"] == 2
    assert state["active"] == 0
    assert persisted_reasons == [
        "task_timeout_exceeded",
        "process_cleanup_failure",
        "task_timeout_exceeded",
        "process_cleanup_failure",
    ]


def test_supervisor_memory_violation_stops_before_dispatch(tmp_path, monkeypatch):
    persisted = []
    monkeypatch.setattr(runner, "_persist_runtime_state", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner,
        "_persist_supervisor_failure",
        lambda *_a, **_k: persisted.append(True),
    )
    policy = projection_preflight_resource_policy("cpu_parallel_1")
    with pytest.raises(P7C4B2CError, match="memory_limit_exceeded"):
        runner._run_supervised_projection_phase(
            [{"sample_id": "one"}],
            manifest={
                "run_id": "fixture-run",
                "worker_count": 1,
                "execution_class": "target_preflight",
                "plan_digest": "a" * 64,
                "authorization_provenance": {"git_commit": "b" * 40},
            },
            environment={},
            run_dir=tmp_path,
            repo_root=tmp_path,
            provenance={
                "execution_stage": TARGET_PROJECTION_PREFLIGHT_STAGE,
                "resource_policy": policy,
            },
            runtime_state={
                "failed_task_count": 0,
                "accumulated_elapsed_seconds": 0.0,
                "last_accounted_at": "2026-08-13T00:00:00Z",
            },
            target_task_gate=lambda: None,
            authorization_deadline=lambda: 1_000_000.0,
            monotonic_clock=lambda: 0.0,
            rss_sampler=lambda _pid: PROJECTION_AGGREGATE_RSS_LIMIT_BYTES + 1,
        )
    assert persisted == [True]


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("environment_lock_hash", "0" * 64, "environment_lock_mismatch"),
        ("free_disk_bytes", MINIMUM_FREE_DISK_BYTES - 1, "insufficient_free_disk"),
        ("output_directory", ".", "unsafe_output_namespace"),
        ("worker_count", 99, "worker_count_mismatch"),
        ("execution_mode", "unsupported", "execution_mode_unsupported"),
        ("hourly_price", float("nan"), "invalid_environment_value"),
        ("hourly_price", float("inf"), "invalid_environment_value"),
        ("price_observed_at", "not-a-timestamp", "invalid_environment_value"),
        ("currency", "US", "invalid_environment_value"),
        ("vm_count", True, "invalid_environment_value"),
        ("maximum_runtime_hours", -1.0, "invalid_environment_value"),
    ],
)
def test_stage_zero_rejects_invalid_evidence(tmp_path, monkeypatch, field, value, code):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    environment[field] = value
    environment["environment_digest"] = environment_digest(environment)
    assert (
        code
        in validate_target_environment(
            environment, plan, repo_root=root, spawn_probe=_pass_probe
        )["reason_codes"]
    )


def test_stage_zero_rejects_missing_inputs_collision_and_spawn_failure(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    (root / "artifacts/p7c4b2d-fixture").mkdir()
    (root / "artifacts/p7c4b2d-fixture/existing.json").write_text(
        "{}", encoding="utf-8"
    )
    report = validate_target_environment(
        environment, plan, repo_root=root, spawn_probe=lambda: {"status": "timeout"}
    )
    assert {"output_collision", "process_spawn_probe_timeout"} <= set(
        report["reason_codes"]
    )
    failed_probe = validate_target_environment(
        environment,
        plan,
        repo_root=root,
        spawn_probe=lambda: {"status": "failed"},
    )
    assert "unsupported_process_spawn" in failed_probe["reason_codes"]
    environment["dataset_hashes"].pop("AC")
    environment["environment_digest"] = environment_digest(environment)
    assert (
        "dataset_input_hash_mismatch"
        in validate_target_environment(
            environment, plan, repo_root=root, spawn_probe=_pass_probe
        )["reason_codes"]
    )


def test_canary_and_proposal_are_mode_complete_non_effective_and_tamper_evident(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    for mode in ("cpu_parallel_1", "cpu_parallel_2"):
        environment = _environment(plan, root, monkeypatch, mode=mode)
        canary = select_canary(plan, mode)
        assert canary["task_count"] == len(canary["task_ids"]) == 4
        proposal = render_authorization_proposal(
            plan, environment, execution_stage="target_canary", expiry=None
        )
        assert (
            proposal["task_ids"] == canary["task_ids"]
            and proposal["authorization_effective"] is False
        )
        assert validate_authorization_proposal(proposal, plan, environment)["valid"]
        proposal["task_ids"] = proposal["task_ids"][:1]
        assert (
            "authorization_proposal_task_scope_mismatch"
            in validate_authorization_proposal(proposal, plan, environment)[
                "reason_codes"
            ]
        )


def test_ready_review_is_not_execution_authorization(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    package = decision_package(
        plan, environment, repo_root=root, canary_complete=True, canary_approved=True
    )
    assert package["readiness"] == "READY_FOR_CANARY_AUTHORIZATION_REVIEW"
    assert package["execution_plan_eligible"] is False
    assert (
        package["canary"]["cpu_parallel_1"]["scientific_projection_eligible"] is False
    )


def test_cli_review_without_environment_is_blocked_without_workload(capsys):
    assert cli_main(["review-plan"]) == EXIT_REVIEW_BLOCKED
    payload = json.loads(capsys.readouterr().out)
    assert "missing_target_environment_metadata" in payload["reason_codes"]


def test_digest_is_deterministic_under_mapping_order(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    reordered = dict(reversed(list(environment.items())))
    assert environment_digest(environment) == environment_digest(reordered)


def test_stage_zero_rejects_git_unknown_and_mismatch(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    monkeypatch.setattr("creditrep.protocols.p7c4b2d.current_git_head", lambda _: None)
    assert (
        "git_provenance_unknown"
        in validate_target_environment(
            environment, plan, repo_root=root, spawn_probe=_pass_probe
        )["reason_codes"]
    )
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.current_git_head", lambda _: "b" * 40
    )
    assert (
        "git_provenance_mismatch"
        in validate_target_environment(
            environment, plan, repo_root=root, spawn_probe=_pass_probe
        )["reason_codes"]
    )


def test_stage_zero_rejects_missing_dataset_file_and_each_required_dataset(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    for missing in ("AC", "GMC"):
        environment = _environment(plan, root, monkeypatch)
        environment["dataset_hashes"].pop(missing)
        environment["environment_digest"] = environment_digest(environment)
        assert (
            "dataset_input_hash_mismatch"
            in validate_target_environment(
                environment, plan, repo_root=root, spawn_probe=_pass_probe
            )["reason_codes"]
        )
    environment = _environment(plan, root, monkeypatch)
    (root / "data/raw/gmc/gmc.csv").unlink()
    assert (
        "dataset_input_missing"
        in validate_target_environment(
            environment, plan, repo_root=root, spawn_probe=_pass_probe
        )["reason_codes"]
    )


def _redigest_proposal(proposal):
    proposal["proposal_digest"] = canonical_digest(proposal, "proposal_digest")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong_mode"])
def test_proposal_rejects_all_task_scope_tampering(tmp_path, monkeypatch, mutation):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    proposal = render_authorization_proposal(
        plan, environment, execution_stage="target_canary", expiry=None
    )
    if mutation == "missing":
        proposal["task_ids"] = proposal["task_ids"][:-1]
    elif mutation == "duplicate":
        proposal["task_ids"][-1] = proposal["task_ids"][0]
    elif mutation == "extra":
        proposal["task_ids"].append(plan["tasks"][0]["sample_id"])
    else:
        proposal["task_ids"] = select_canary(plan, "cpu_parallel_2")["task_ids"]
    _redigest_proposal(proposal)
    assert (
        "authorization_proposal_task_scope_mismatch"
        in validate_authorization_proposal(proposal, plan, environment)["reason_codes"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("plan_digest", "0" * 64),
        ("target_environment_digest", "0" * 64),
        ("output_directory", "artifacts/other"),
        ("maximum_runtime_hours", 99.0),
        ("maximum_monetary_budget", 99.0),
    ],
)
def test_proposal_rejects_bound_scope_mismatches(tmp_path, monkeypatch, field, value):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    proposal = render_authorization_proposal(
        plan, environment, execution_stage="target_canary", expiry=None
    )
    proposal[field] = value
    _redigest_proposal(proposal)
    assert (
        "authorization_mismatch"
        in validate_authorization_proposal(proposal, plan, environment)["reason_codes"]
    )


def test_expired_or_effective_proposal_is_rejected(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    proposal = render_authorization_proposal(
        plan,
        environment,
        execution_stage="target_canary",
        expiry="2000-01-01T00:00:00Z",
    )
    assert (
        "authorization_expired"
        in validate_authorization_proposal(proposal, plan, environment)["reason_codes"]
    )
    proposal = render_authorization_proposal(
        plan, environment, execution_stage="target_canary", expiry=None
    )
    proposal["authorization_effective"] = True
    _redigest_proposal(proposal)
    assert (
        "authorization_proposal_invalid"
        in validate_authorization_proposal(proposal, plan, environment)["reason_codes"]
    )


def test_cli_review_uses_valid_environment_and_remains_non_executing(
    tmp_path, monkeypatch, capsys
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(environment), encoding="utf-8")
    monkeypatch.setattr(cli, "find_repo_root", lambda: root)
    monkeypatch.setattr(cli, "_plan", lambda _: plan)
    assert cli.main(["review-plan", "--environment", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness"] == "READY_FOR_CANARY_AUTHORIZATION_REVIEW"
    assert payload["execution_plan_eligible"] is False


def test_operator_metadata_merges_valid_schema_and_recomputes_digest(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    collected = _environment(plan, root, monkeypatch)
    for field in _operator_metadata():
        collected[field] = None
    collected["environment_digest"] = environment_digest(collected)
    merged = merge_operator_metadata(collected, _operator_metadata())
    assert merged["environment_digest"] == environment_digest(merged)
    assert validate_target_environment(
        merged, plan, repo_root=root, spawn_probe=_pass_probe
    )["valid"]


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda value: value.pop("provider"), "operator_metadata_missing_field"),
        (
            lambda value: value.__setitem__("unknown", "x"),
            "operator_metadata_unknown_field",
        ),
        (
            lambda value: value.__setitem__("vm_count", True),
            "operator_metadata_invalid",
        ),
        (
            lambda value: value.__setitem__("hourly_price", float("nan")),
            "operator_metadata_invalid",
        ),
        (
            lambda value: value.__setitem__("git_commit", "b" * 40),
            "operator_metadata_canonical_override",
        ),
    ],
)
def test_operator_metadata_is_strict_and_cannot_override_canonical(
    tmp_path, monkeypatch, mutation, code
):
    plan, root = _plan(), _repo(tmp_path)
    collected = _environment(plan, root, monkeypatch)
    metadata = _operator_metadata()
    mutation(metadata)
    with pytest.raises(P7C4B2DError, match=code):
        merge_operator_metadata(collected, metadata)


def test_operator_metadata_digest_is_stable_and_tampering_is_rejected(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    collected = _environment(plan, root, monkeypatch)
    first = merge_operator_metadata(collected, _operator_metadata())
    second = merge_operator_metadata(
        dict(reversed(list(collected.items()))), _operator_metadata()
    )
    assert first["environment_digest"] == second["environment_digest"]
    first["maximum_monetary_budget"] = 99.0
    assert (
        "invalid_environment_value"
        in validate_target_environment(
            first, plan, repo_root=root, spawn_probe=_pass_probe
        )["reason_codes"]
    )


def test_collector_measures_cpu_ram_and_merges_only_operator_fields(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.current_git_head", lambda _: "a" * 40
    )
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.collect_dataset_hashes",
        lambda *_args, **_kwargs: {"AC": "A" * 64, "GMC": "B" * 64},
    )
    monkeypatch.setattr("creditrep.protocols.p7c4b2d.probe_process_spawn", _pass_probe)
    monkeypatch.setattr("creditrep.protocols.p7c4b2d.os.cpu_count", lambda: 12)
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.psutil.virtual_memory",
        lambda: SimpleNamespace(total=987654321),
    )
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.platform.processor", lambda: "fixture-cpu"
    )
    value = collect_target_environment(
        plan,
        mode="cpu_parallel_1",
        output_directory="artifacts/p7c4b2d-fixture",
        operator_metadata=_operator_metadata(),
        repo_root=root,
    )
    assert (value["vcpu_count"], value["ram_bytes"], value["cpu_model"]) == (
        12,
        987654321,
        "fixture-cpu",
    )
    assert value["provider"] == "fixture-cloud"
    assert value["environment_digest"] == environment_digest(value)


@pytest.mark.parametrize(
    "relative_output,prepare,expected_anchor",
    [
        ("artifacts/existing-target", "target", "artifacts/existing-target"),
        ("artifacts/existing-parent/target", "parent", "artifacts/existing-parent"),
        ("artifacts/missing/a/b/target", "none", "artifacts"),
    ],
)
def test_collector_uses_nearest_existing_filesystem_anchor_without_creating_output(
    tmp_path, monkeypatch, relative_output, prepare, expected_anchor
):
    plan, root = _plan(), _repo(tmp_path)
    target = root / relative_output
    if prepare == "target":
        target.mkdir()
    elif prepare == "parent":
        target.parent.mkdir()
    observed = []
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.shutil.disk_usage",
        lambda path: (observed.append(Path(path)) or SimpleNamespace(free=123456)),
    )
    value = collect_target_environment(
        plan,
        mode="cpu_parallel_1",
        output_directory=relative_output,
        repo_root=root,
    )
    assert value["free_disk_bytes"] == 123456
    assert observed == [root / expected_anchor]
    assert target.exists() is (prepare == "target")


def test_collector_filesystem_probe_error_fails_closed(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    target = root / "artifacts" / "missing" / "target"
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.shutil.disk_usage",
        lambda _path: (_ for _ in ()).throw(OSError("fixture failure")),
    )
    value = collect_target_environment(
        plan,
        mode="cpu_parallel_1",
        output_directory=str(target),
        repo_root=root,
    )
    assert value["free_disk_bytes"] is None
    assert not target.exists()


def test_collector_uses_symlinked_filesystem_anchor(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    link = tmp_path / "artifact-link"
    try:
        link.symlink_to(root / "artifacts", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    observed = []
    monkeypatch.setattr(
        "creditrep.protocols.p7c4b2d.shutil.disk_usage",
        lambda path: (observed.append(Path(path)) or SimpleNamespace(free=123456)),
    )
    value = collect_target_environment(
        plan,
        mode="cpu_parallel_1",
        output_directory=str(link / "missing" / "target"),
        repo_root=root,
    )
    assert value["free_disk_bytes"] == 123456
    assert len(observed) == 1
    assert observed[0].resolve() == (root / "artifacts").resolve()
    assert not (link / "missing" / "target").exists()


def test_cli_collect_metadata_and_complete_evidence_review_remain_non_effective(
    tmp_path, monkeypatch, capsys
):
    plan, root = _plan(), _repo(tmp_path)
    metadata_path = tmp_path / "operator-metadata.json"
    metadata_path.write_text(json.dumps(_operator_metadata()), encoding="utf-8")
    merged = _environment(plan, root, monkeypatch)
    monkeypatch.setattr(cli, "find_repo_root", lambda: root)
    monkeypatch.setattr(cli, "_plan", lambda _: plan)
    monkeypatch.setattr(
        cli,
        "collect_target_environment",
        lambda *_args, **kwargs: merge_operator_metadata(
            merged, kwargs["operator_metadata"]
        ),
    )
    assert (
        cli.main(
            [
                "collect-target-environment",
                "--output-directory",
                "artifacts/p7c4b2d-fixture",
                "--operator-metadata",
                str(metadata_path),
            ]
        )
        == EXIT_REVIEW_BLOCKED
    )
    collected = json.loads(capsys.readouterr().out)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(collected), encoding="utf-8")
    assert (
        cli.main(["inspect-target-requirements", "--environment", str(evidence_path)])
        == 0
    )
    capsys.readouterr()
    assert cli.main(["review-plan", "--environment", str(evidence_path)]) == 0
    review = json.loads(capsys.readouterr().out)
    assert review["execution_plan_eligible"] is False


def test_projection_preflight_cli_json_round_trip_reaches_non_effective_proposal(
    tmp_path, monkeypatch, capsys
):
    plan, root = _plan(), _repo(tmp_path)
    metadata_path = tmp_path / "operator-metadata.json"
    metadata = _operator_metadata()
    metadata.update(
        {
            "maximum_runtime_hours": 12,
            "maximum_monetary_budget": "5.0",
            "hourly_price": "0.26",
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    collected = _as_projection_environment(
        _environment(plan, root, monkeypatch), plan, root
    )
    monkeypatch.setattr(cli, "find_repo_root", lambda: root)
    monkeypatch.setattr(cli, "_plan", lambda _: plan)
    monkeypatch.setattr(
        cli,
        "collect_target_environment",
        lambda *_args, **kwargs: merge_operator_metadata(
            collected, kwargs["operator_metadata"]
        ),
    )
    monkeypatch.setattr("creditrep.protocols.p7c4b2d.probe_process_spawn", _pass_probe)

    assert cli.main(
        [
            "collect-target-environment",
            "--mode",
            "cpu_parallel_1",
            "--stage",
            TARGET_PROJECTION_PREFLIGHT_STAGE,
            "--output-directory",
            "artifacts/projection-preflight-fixture",
            "--operator-metadata",
            str(metadata_path),
        ]
    ) == EXIT_REVIEW_BLOCKED
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(capsys.readouterr().out, encoding="utf-8")

    assert cli.main(["inspect-target-requirements", "--environment", str(environment_path)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["valid"]

    assert cli.main(
        [
            "render-authorization-proposal",
            "--stage",
            TARGET_PROJECTION_PREFLIGHT_STAGE,
            "--environment",
            str(environment_path),
        ]
    ) == 0
    proposal = json.loads(capsys.readouterr().out)
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal, sort_keys=True), encoding="utf-8")
    assert proposal["maximum_task_count"] == 162
    assert proposal["authorization_effective"] is False
    assert cli.main(
        [
            "validate-authorization-proposal",
            "--environment",
            str(environment_path),
            "--proposal",
            str(proposal_path),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_effective_authorization_is_explicit_scoped_and_valid(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _proposal_and_authorization(
        plan, root, monkeypatch
    )
    assert authorization["artifact_type"] == "effective_authorization"
    assert authorization["authorization_effective"] is True
    assert (
        authorization["task_ids"] == select_canary(plan, "cpu_parallel_1")["task_ids"]
    )
    report = validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=datetime.fromisoformat("2026-08-10T00:30:00+00:00"),
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert report["valid"] is True


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"operator_identity": ""}, "operator_identity_missing"),
        ({"operator_approval": "wrong"}, "operator_approval_missing"),
        ({"expires_at": "2026-08-10T00:00:00Z"}, "expiry_missing_or_invalid"),
        ({"expires_at": "2026-08-12T00:00:00Z"}, "expiry_missing_or_invalid"),
    ],
)
def test_effective_authorization_creation_rejects_missing_or_invalid_inputs(
    tmp_path, monkeypatch, kwargs, code
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    proposal = render_authorization_proposal(
        plan, environment, execution_stage="target_canary", expiry=None
    )
    values = {
        "operator_identity": "fixture-operator",
        "operator_approval": TARGET_CANARY_APPROVAL,
        "expires_at": "2026-08-10T01:00:00Z",
    }
    values.update(kwargs)
    with pytest.raises(P7C4B2DError, match=code):
        _create_effective_authorization(
            proposal,
            environment,
            plan,
            repo_root=root,
            spawn_probe=_pass_probe,
            now_provider=lambda: datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
            **values,
        )


def test_effective_authorization_requires_budget_for_full_authorized_window(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment = _environment(plan, root, monkeypatch)
    environment["maximum_monetary_budget"] = "24.99"
    environment["environment_digest"] = environment_digest(environment)
    proposal = render_authorization_proposal(
        plan, environment, execution_stage="target_canary", expiry=None
    )
    with pytest.raises(P7C4B2DError, match="monetary_budget_mismatch"):
        _create_effective_authorization(
            proposal,
            environment,
            plan,
            operator_identity="fixture-operator",
            operator_approval=TARGET_CANARY_APPROVAL,
            expires_at="2026-08-10T01:00:00Z",
            repo_root=root,
            spawn_probe=_pass_probe,
            now_provider=lambda: datetime.fromisoformat("2026-08-10T00:00:00+00:00"),
        )


@pytest.mark.parametrize(
    "mutation,code",
    [
        (
            lambda value: value.__setitem__("artifact_type", "authorization_proposal"),
            "wrong_artifact_type",
        ),
        (
            lambda value: value.__setitem__("authorization_effective", False),
            "authorization_not_effective",
        ),
        (
            lambda value: value.__setitem__("operator_identity", ""),
            "operator_identity_missing",
        ),
        (lambda value: value.__setitem__("task_ids", []), "task_scope_mismatch"),
        (
            lambda value: value.__setitem__("execution_mode", "cpu_parallel_2"),
            "execution_mode_mismatch",
        ),
        (
            lambda value: value.__setitem__("output_directory", "artifacts/other"),
            "output_directory_mismatch",
        ),
        (
            lambda value: value.__setitem__("maximum_runtime_hours", 99.0),
            "runtime_limit_mismatch",
        ),
        (
            lambda value: value.__setitem__("maximum_monetary_budget", 99.0),
            "monetary_budget_mismatch",
        ),
    ],
)
def test_effective_authorization_rejects_tampering_and_scope_mismatch(
    tmp_path, monkeypatch, mutation, code
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _proposal_and_authorization(
        plan, root, monkeypatch
    )
    mutation(authorization)
    report = validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=datetime.fromisoformat("2026-08-10T00:30:00+00:00"),
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert code in report["reason_codes"]
    assert "authorization_digest_mismatch" in report["reason_codes"]


def test_recomputed_digest_cannot_expand_effective_authorization_scope(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _proposal_and_authorization(
        plan, root, monkeypatch
    )
    authorization["task_ids"] = []
    authorization["authorization_digest"] = canonical_digest(
        authorization, "authorization_digest"
    )
    report = validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=datetime.fromisoformat("2026-08-10T00:30:00+00:00"),
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert "authorization_digest_mismatch" not in report["reason_codes"]
    assert "task_scope_mismatch" in report["reason_codes"]


def test_effective_authorization_rejects_non_finite_tampering_fail_closed(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _proposal_and_authorization(
        plan, root, monkeypatch
    )
    authorization["maximum_monetary_budget"] = float("nan")
    report = validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=datetime.fromisoformat("2026-08-10T00:30:00+00:00"),
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert "authorization_digest_mismatch" in report["reason_codes"]
    assert "monetary_budget_mismatch" in report["reason_codes"]


def test_effective_authorization_rejects_missing_proposal_and_expiry(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _proposal_and_authorization(
        plan, root, monkeypatch
    )
    missing = validate_effective_authorization(
        None, proposal, environment, plan, repo_root=root, spawn_probe=_pass_probe
    )
    assert missing["reason_codes"] == ["authorization_missing"]
    proposal_as_auth = validate_effective_authorization(
        proposal, proposal, environment, plan, repo_root=root, spawn_probe=_pass_probe
    )
    assert {"wrong_artifact_type", "authorization_not_effective"} <= set(
        proposal_as_auth["reason_codes"]
    )
    expired = validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=datetime.fromisoformat("2026-08-10T02:00:00+00:00"),
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert "authorization_expired" in expired["reason_codes"]


def test_public_creation_api_has_no_created_at_override():
    import inspect

    assert (
        "created_at" not in inspect.signature(create_effective_authorization).parameters
    )


@pytest.mark.parametrize(
    "artifact,field,code",
    [
        ("proposal", "schema_version", "authorization_proposal_schema_mismatch"),
        ("proposal", "checkpoint", "authorization_proposal_checkpoint_mismatch"),
        ("authorization", "schema_version", "authorization_schema_mismatch"),
        ("authorization", "checkpoint", "authorization_checkpoint_mismatch"),
    ],
)
def test_schema_and_checkpoint_are_closed_world(
    tmp_path, monkeypatch, artifact, field, code
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _proposal_and_authorization(
        plan, root, monkeypatch
    )
    value = proposal if artifact == "proposal" else authorization
    value[field] = 999 if field == "schema_version" else "foreign"
    digest_field = (
        "proposal_digest" if artifact == "proposal" else "authorization_digest"
    )
    value[digest_field] = canonical_digest(value, digest_field)
    report = (
        validate_authorization_proposal(proposal, plan, environment)
        if artifact == "proposal"
        else validate_effective_authorization(
            authorization,
            proposal,
            environment,
            plan,
            now=datetime.fromisoformat("2026-08-10T00:30:00+00:00"),
            repo_root=root,
            spawn_probe=_pass_probe,
        )
    )
    assert code in report["reason_codes"]


def test_decimal_budget_boundaries_are_exact():
    base = {"hourly_price": "0.26092", "vm_count": 1, "maximum_runtime_hours": 12}
    assert _static_authorization_cost_upper_bound(base) == __import__(
        "decimal"
    ).Decimal("3.13104")
    for budget, allowed in (
        ("3.13104", True),
        ("3.1310400001", True),
        ("3.1310399999", False),
    ):
        upper = _static_authorization_cost_upper_bound(base)
        assert (__import__("decimal").Decimal(budget) >= upper) is allowed


def test_output_normalization_rejects_traversal_and_symlink_escape(tmp_path):
    root = tmp_path
    artifacts = root / "artifacts"
    artifacts.mkdir()
    exact = artifacts / "target"
    assert normalize_target_output("artifacts/target", root) == str(exact.resolve())
    assert normalize_target_output(exact, root) == str(exact.resolve())
    with pytest.raises(P7C4B2DError, match="unsafe_output_namespace"):
        normalize_target_output("artifacts/../outside", root)
    outside = root / "outside"
    outside.mkdir()
    link = artifacts / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(P7C4B2DError, match="unsafe_output_namespace"):
        normalize_target_output(link, root)


@pytest.mark.parametrize(
    "field,value,expected_valid",
    [
        ("operator_identity", "different-operator", True),
        ("operator_approval", "different-approval", False),
        ("created_at", "2026-08-10T00:05:00Z", True),
        ("expires_at", "2026-08-10T00:55:00Z", True),
    ],
)
def test_recomputed_checksum_only_fields_follow_semantic_not_signature_rules(
    tmp_path, monkeypatch, field, value, expected_valid
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _proposal_and_authorization(
        plan, root, monkeypatch
    )
    authorization[field] = value
    authorization["authorization_digest"] = canonical_digest(
        authorization, "authorization_digest"
    )
    report = validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        now=datetime.fromisoformat("2026-08-10T00:30:00+00:00"),
        repo_root=root,
        spawn_probe=_pass_probe,
    )
    assert report["valid"] is expected_valid
    assert "authorization_digest_mismatch" not in report["reason_codes"]


@pytest.mark.parametrize(
    "payload,code",
    [
        ("{", "malformed_json"),
        ('{"a": 1, "a": 2}', "duplicate_json_key"),
        ('{"nested": {"a": 1, "a": 2}}', "duplicate_json_key"),
        ('{"value": NaN}', "invalid_json_number"),
        ('{"value": Infinity}', "invalid_json_number"),
        ("[]", "invalid_json_object"),
        ('"text"', "invalid_json_object"),
        ("null", "invalid_json_object"),
    ],
)
def test_cli_strict_json_failures_are_stable(
    tmp_path, monkeypatch, capsys, payload, code
):
    root = _repo(tmp_path)
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(cli, "find_repo_root", lambda: root)
    monkeypatch.setattr(cli, "_plan", lambda _: _plan())
    assert cli.main(["review-plan", "--environment", str(path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["reason_codes"] == [code]


class _DispatchReached(RuntimeError):
    pass


def _fresh_runner_authorization(plan, root, monkeypatch):
    environment = _environment(plan, root, monkeypatch)
    proposal = render_authorization_proposal(
        plan, environment, execution_stage="target_canary", expiry=None
    )
    now = datetime.now(UTC)
    authorization = _create_effective_authorization(
        proposal,
        environment,
        plan,
        operator_identity="fixture-operator",
        operator_approval=TARGET_CANARY_APPROVAL,
        expires_at=(now + timedelta(hours=1)).isoformat(),
        repo_root=root,
        spawn_probe=_pass_probe,
        now_provider=lambda: now,
    )
    return environment, proposal, authorization


def _stop_before_worker(monkeypatch):
    monkeypatch.setattr(
        runner,
        "ProcessPoolExecutor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_DispatchReached()),
    )


def _stopped_target_run(tmp_path, monkeypatch):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _fresh_runner_authorization(
        plan, root, monkeypatch
    )
    output = root / environment["output_directory"]
    monkeypatch.setattr(
        runner,
        "capture_environment",
        lambda _root: {
            "schema_version": 2,
            "git_commit": "a" * 40,
            "environment_digest": "fixture",
        },
    )
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_FREE_DISK_BYTES + 1),
    )
    _stop_before_worker(monkeypatch)
    with pytest.raises(_DispatchReached):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    return plan, root, output, environment, proposal, authorization


def _complete_four_task_target_fixture(tmp_path, monkeypatch):
    plan, root, output, environment, proposal, authorization = _stopped_target_run(
        tmp_path, monkeypatch
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    persisted_environment = json.loads(
        (output / "environment.json").read_text(encoding="utf-8")
    )
    persisted_environment["schema_version"] = runner.SCHEMA_VERSION
    persisted_environment["environment_digest"] = runner.canonical_digest(
        persisted_environment, "environment_digest"
    )
    runner._atomic_json(output / "environment.json", persisted_environment)
    tasks = manifest["expected_tasks"]
    assert len(tasks) == 4
    for index, task in enumerate(tasks):
        record = {
            **task,
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "attempt": 1,
            "attempt_id": f"fixture-attempt-{index}",
            "execution_class": "target_preflight",
            "git_commit": persisted_environment["git_commit"],
            "plan_digest": plan["plan_digest"],
            "preprocessing_identity": "fixture",
            "input_identity": {},
            "input_hash": runner.sha256_canonical({}),
            "started_utc": "2026-08-10T00:00:00Z",
            "completed_utc": "2026-08-10T00:00:00Z",
            "started_monotonic": 0.0,
            "completed_monotonic": 0.0,
            "status": "completed",
            "projection_eligible": False,
            "limitations": ["synthetic_target_canary_fixture"],
            "result": {},
            **{field: 0.0 for field in TIMING_FIELDS},
        }
        sample_dir = output / "samples" / task["sample_id"]
        sample_dir.mkdir(parents=True)
        runner._atomic_json(sample_dir / "result.json", record)
        runner._atomic_json(sample_dir / "telemetry.json", record)
        runner._atomic_json(
            sample_dir / "COMPLETED.json",
            {
                "record_digest": runner.sha256_canonical(record),
                "attempt_id": record["attempt_id"],
            },
        )
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "samples").glob("*/result.json"))
    ]
    coverage, summary = runner.summarize(
        records, plan, expected_tasks=manifest["expected_tasks"]
    )
    runner._atomic_json(output / "coverage.json", coverage)
    runner._atomic_json(output / "stratum_summary.json", summary)
    preliminary = runner._validate_artifacts(
        output, allow_missing_derived=True, allow_missing_marker=True
    )
    projection = runner.project_validated(
        records,
        plan,
        artifact_validation=preliminary,
        execution_class="target_preflight",
    )
    runner._atomic_json(output / "projection.json", projection)
    runner._atomic_json(
        output / "eligibility.json",
        {
            "schema_version": runner.SCHEMA_VERSION,
            "execution_plan_eligible": projection["execution_plan_eligible"],
            "reason_codes": projection["reason_codes"],
        },
    )
    report = runner._validate_artifacts(output, allow_missing_marker=True)
    runner._atomic_json(output / "validation.json", report)
    runner._atomic_json(
        output / "COMPLETED.json",
        {
            "validation_digest": runner.sha256_canonical(report),
            "run_id": manifest["run_id"],
        },
    )
    return plan, root, output, environment, proposal, authorization


def _finalize_completed_fixture_as_initial_run(
    output, root, environment, proposal, authorization
):
    (output / "COMPLETED.json").unlink()
    return runner._resume_impl(
        output,
        repo_root=root,
        target_environment=environment,
        authorization_proposal=proposal,
        effective_authorization=authorization,
        target_authorized=False,
        authorization_plan_digest=None,
        wall_clock=lambda: datetime.now(UTC),
        monotonic_clock=time.perf_counter,
        disk_usage_provider=runner.shutil.disk_usage,
        initial_authorization_validated=True,
    )


def test_four_task_target_canary_acceptance_is_separate_from_scientific_coverage(
    tmp_path, monkeypatch
):
    _plan_value, _root, output, *_authorization = _complete_four_task_target_fixture(
        tmp_path, monkeypatch
    )
    report = runner.validate_artifacts(output)
    projection = json.loads((output / "projection.json").read_text(encoding="utf-8"))
    coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["completed"] == report["expected"] == 4
    assert report["target_canary_acceptance"]["applicable"] is True
    assert report["target_canary_acceptance"]["accepted"] is True
    assert report["target_canary_acceptance"]["reason_codes"] == []
    assert report["target_canary_acceptance"]["scientific_projection_eligible"] is False
    assert (
        report["target_canary_acceptance"]["canonical_scientific_execution_authorized"]
        is False
    )
    assert report["scientific_coverage"]["valid"] is False
    assert report["scientific_coverage"]["reason_codes"] == [
        "incomplete_required_stratum",
        "insufficient_repetitions",
    ]
    assert coverage["minimum_repetitions"] == 2
    assert coverage["covered_strata"] == 0
    assert projection["execution_plan_eligible"] is False
    assert "incomplete_required_stratum" in projection["reason_codes"]
    assert "insufficient_repetitions" in projection["reason_codes"]
    assert (
        preflight_cli_main(["validate-artifacts", "--run-dir", str(output)])
        == runner.EXIT_OK
    )
    assert (
        preflight_cli_main(["project", "--run-dir", str(output)])
        == runner.EXIT_INCOMPLETE
    )


def test_fresh_target_missing_run_marker_fails_normal_validation_and_cli(
    tmp_path, monkeypatch
):
    _plan_value, _root, output, *_authorization = _complete_four_task_target_fixture(
        tmp_path, monkeypatch
    )
    (output / "COMPLETED.json").unlink()

    report = runner.validate_artifacts(output)

    assert report["valid"] is False
    assert report["target_canary_acceptance"]["accepted"] is False
    assert "completed_run_missing_marker" in report["reason_codes"]
    assert (
        preflight_cli_main(["validate-artifacts", "--run-dir", str(output)])
        == runner.EXIT_VALIDATION
    )


def test_legacy_shaped_validation_cannot_bypass_missing_run_marker(
    tmp_path, monkeypatch
):
    _plan_value, _root, output, *_authorization = _complete_four_task_target_fixture(
        tmp_path, monkeypatch
    )
    (output / "COMPLETED.json").unlink()
    runner._atomic_json(
        output / "validation.json",
        {
            "schema_version": runner.SCHEMA_VERSION,
            "valid": False,
            "reason_codes": [
                "incomplete_required_stratum",
                "insufficient_repetitions",
            ],
            "expected": 4,
            "completed": 4,
        },
    )

    report = runner.validate_artifacts(output)

    assert report["valid"] is False
    assert report["target_canary_acceptance"]["accepted"] is False
    assert "completed_run_missing_marker" in report["reason_codes"]
    assert (
        preflight_cli_main(["validate-artifacts", "--run-dir", str(output)])
        == runner.EXIT_VALIDATION
    )


@pytest.mark.parametrize(
    "artifact,value",
    [
        ("environment", "b" * 40),
        ("environment", None),
        ("environment", "NOT-A-LOWERCASE-SHA40"),
        ("sample", "b" * 40),
        ("sample", None),
        ("sample", "NOT-A-LOWERCASE-SHA40"),
        ("provenance", "b" * 40),
        ("provenance", None),
        ("provenance", "NOT-A-LOWERCASE-SHA40"),
    ],
)
def test_target_source_identity_is_cross_bound_and_strictly_validated(
    tmp_path, monkeypatch, artifact, value
):
    _plan_value, _root, output, *_authorization = _complete_four_task_target_fixture(
        tmp_path, monkeypatch
    )
    if artifact == "environment":
        environment_path = output / "environment.json"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        if value is None:
            environment.pop("git_commit")
        else:
            environment["git_commit"] = value
        environment["environment_digest"] = runner.canonical_digest(
            environment, "environment_digest"
        )
        runner._atomic_json(environment_path, environment)
    elif artifact == "sample":
        sample_dir = next((output / "samples").iterdir())
        record_path = sample_dir / "result.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if value is None:
            record.pop("git_commit")
        else:
            record["git_commit"] = value
        runner._atomic_json(record_path, record)
        runner._atomic_json(sample_dir / "telemetry.json", record)
        runner._atomic_json(
            sample_dir / "COMPLETED.json",
            {
                "record_digest": runner.sha256_canonical(record),
                "attempt_id": record["attempt_id"],
            },
        )
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output / "samples").glob("*/result.json"))
        ]
        projection_path = output / "projection.json"
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        projection["source_evidence_digest"] = runner.sha256_canonical(records)
        runner._atomic_json(projection_path, projection)
    else:
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if value is None:
            manifest["authorization_provenance"].pop("git_commit")
        else:
            manifest["authorization_provenance"]["git_commit"] = value
        runner._atomic_json(manifest_path, manifest)

    report = runner.validate_artifacts(output)

    assert report["valid"] is False
    assert report["target_canary_acceptance"]["accepted"] is False
    assert "git_provenance_mismatch" in report["reason_codes"]


def test_target_output_path_is_bound_to_persisted_authorization(tmp_path, monkeypatch):
    _plan_value, _root, output, *_authorization = _complete_four_task_target_fixture(
        tmp_path, monkeypatch
    )
    moved = output.with_name(f"{output.name}-moved")
    output.rename(moved)

    report = runner.validate_artifacts(moved)

    assert report["valid"] is False
    assert report["target_canary_acceptance"]["accepted"] is False
    assert "authorization_provenance_mismatch" in report["reason_codes"]


def test_target_run_completion_marker_is_bound_to_manifest_run_id(
    tmp_path, monkeypatch
):
    _plan_value, _root, output, *_authorization = _complete_four_task_target_fixture(
        tmp_path, monkeypatch
    )
    marker_path = output / "COMPLETED.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["run_id"] = "different-run"
    runner._atomic_json(marker_path, marker)

    report = runner.validate_artifacts(output)

    assert report["valid"] is False
    assert report["target_canary_acceptance"]["accepted"] is False
    assert "run_complete_marker_integrity_failure" in report["reason_codes"]


def test_valid_finalization_returns_post_marker_public_validation(
    tmp_path, monkeypatch
):
    _plan_value, root, output, environment, proposal, authorization = (
        _complete_four_task_target_fixture(tmp_path, monkeypatch)
    )
    public_calls = []
    original_validate = runner.validate_artifacts

    def tracked_public_validation(run_dir, **kwargs):
        public_calls.append((run_dir / "COMPLETED.json").exists())
        return original_validate(run_dir, **kwargs)

    monkeypatch.setattr(runner, "validate_artifacts", tracked_public_validation)
    result = _finalize_completed_fixture_as_initial_run(
        output, root, environment, proposal, authorization
    )

    assert public_calls == [True]
    assert result["validation"]["valid"] is True
    assert result["validation"] == original_validate(output)
    assert (
        preflight_cli_main(["validate-artifacts", "--run-dir", str(output)])
        == runner.EXIT_OK
    )


def test_corrupt_marker_write_returns_post_marker_failure_and_cli_exit_two(
    tmp_path, monkeypatch
):
    _plan_value, root, output, environment, proposal, authorization = (
        _complete_four_task_target_fixture(tmp_path, monkeypatch)
    )
    original_write = runner._atomic_json

    def corrupt_run_marker(path, value):
        if path == output / "COMPLETED.json":
            value = {**value, "run_id": "corrupt-run-id"}
        original_write(path, value)

    monkeypatch.setattr(runner, "_atomic_json", corrupt_run_marker)
    result = _finalize_completed_fixture_as_initial_run(
        output, root, environment, proposal, authorization
    )

    assert result["validation"]["valid"] is False
    assert (
        "run_complete_marker_integrity_failure" in result["validation"]["reason_codes"]
    )
    assert (
        preflight_cli_main(["validate-artifacts", "--run-dir", str(output)])
        == runner.EXIT_VALIDATION
    )


def test_marker_write_exception_propagates_without_success(tmp_path, monkeypatch):
    _plan_value, root, output, environment, proposal, authorization = (
        _complete_four_task_target_fixture(tmp_path, monkeypatch)
    )
    original_write = runner._atomic_json

    def fail_run_marker(path, value):
        if path == output / "COMPLETED.json":
            raise OSError("simulated marker write failure")
        original_write(path, value)

    monkeypatch.setattr(runner, "_atomic_json", fail_run_marker)
    with pytest.raises(OSError, match="simulated marker write failure"):
        _finalize_completed_fixture_as_initial_run(
            output, root, environment, proposal, authorization
        )
    assert not (output / "COMPLETED.json").exists()


def test_completed_markerless_resume_is_rejected_without_reclassification(
    tmp_path, monkeypatch, capsys
):
    plan, root, output, environment, proposal, authorization = (
        _complete_four_task_target_fixture(tmp_path, monkeypatch)
    )
    marker_path = output / "COMPLETED.json"
    marker_path.unlink()
    validation_path = output / "validation.json"
    validation_before = validation_path.read_bytes()

    result = runner.resume(
        output,
        repo_root=root,
        target_environment=environment,
        authorization_proposal=proposal,
        effective_authorization=authorization,
    )

    assert result["executed"] == 0
    assert result["skipped"] == 4
    assert result["validation"]["valid"] is False
    assert "completed_run_missing_marker" in result["validation"]["reason_codes"]
    assert not marker_path.exists()
    assert validation_path.read_bytes() == validation_before

    environment_path = tmp_path / "target-environment.json"
    proposal_path = tmp_path / "proposal.json"
    authorization_path = tmp_path / "authorization.json"
    runner._atomic_json(environment_path, environment)
    runner._atomic_json(proposal_path, proposal)
    runner._atomic_json(authorization_path, authorization)
    monkeypatch.setattr(preflight_cli, "find_repo_root", lambda: root)
    monkeypatch.setattr(preflight_cli, "_default_plan", lambda _root: plan)
    assert (
        preflight_cli.main(
            [
                "resume",
                "--run-dir",
                str(output),
                "--target-environment",
                str(environment_path),
                "--authorization-proposal",
                str(proposal_path),
                "--effective-authorization",
                str(authorization_path),
            ]
        )
        == runner.EXIT_VALIDATION
    )
    capsys.readouterr()


def test_resume_rejects_existing_corrupt_run_marker_without_repair(
    tmp_path, monkeypatch
):
    _plan_value, root, output, environment, proposal, authorization = (
        _complete_four_task_target_fixture(tmp_path, monkeypatch)
    )
    marker_path = output / "COMPLETED.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["run_id"] = "corrupt-run-id"
    runner._atomic_json(marker_path, marker)
    marker_before = marker_path.read_bytes()
    validation_path = output / "validation.json"
    validation_before = validation_path.read_bytes()

    result = runner.resume(
        output,
        repo_root=root,
        target_environment=environment,
        authorization_proposal=proposal,
        effective_authorization=authorization,
    )

    assert result["executed"] == 0
    assert result["validation"]["valid"] is False
    assert (
        "run_complete_marker_integrity_failure" in result["validation"]["reason_codes"]
    )
    assert marker_path.read_bytes() == marker_before
    assert validation_path.read_bytes() == validation_before


def test_incomplete_resume_finishes_without_fit_and_uses_post_marker_validation(
    tmp_path, monkeypatch
):
    _plan_value, root, output, environment, proposal, authorization = (
        _complete_four_task_target_fixture(tmp_path, monkeypatch)
    )
    (output / "COMPLETED.json").unlink()
    missing = next((output / "samples").iterdir())
    shutil.rmtree(missing)

    def no_fit_outer_refit(_task, _root, *, locked_runtime_inputs):
        assert locked_runtime_inputs is not None
        return {
            "timings": {field: 0.0 for field in TIMING_FIELDS},
            "preprocessing_identity": "no-fit-resume-fixture",
            "input_identity": {},
            "limitations": ["no_fit_resume_fixture"],
            "result": {},
        }

    class InlineExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover - surfaced by future.result
                future.set_exception(exc)
            return future

    monkeypatch.setattr(runner, "canonical_outer_refit", no_fit_outer_refit)
    monkeypatch.setattr(
        runner, "ProcessPoolExecutor", lambda *_args, **_kwargs: InlineExecutor()
    )
    public_calls = []
    original_validate = runner.validate_artifacts

    def tracked_public_validation(run_dir, **kwargs):
        public_calls.append((run_dir / "COMPLETED.json").exists())
        return original_validate(run_dir, **kwargs)

    monkeypatch.setattr(runner, "validate_artifacts", tracked_public_validation)
    result = runner.resume(
        output,
        repo_root=root,
        target_environment=environment,
        authorization_proposal=proposal,
        effective_authorization=authorization,
    )

    assert result["executed"] == 1
    assert result["skipped"] == 3
    assert public_calls == [True]
    assert result["validation"]["valid"] is True
    assert result["validation"] == original_validate(output)


def test_complete_four_task_target_canary_resume_skips_without_expanding_scope(
    tmp_path, monkeypatch
):
    _plan_value, root, output, environment, proposal, authorization = (
        _complete_four_task_target_fixture(tmp_path, monkeypatch)
    )
    result = runner.resume(
        output,
        repo_root=root,
        target_environment=environment,
        authorization_proposal=proposal,
        effective_authorization=authorization,
    )
    assert result["executed"] == 0
    assert result["skipped"] == 4
    assert result["validation"]["valid"] is True
    assert result["validation"]["scientific_coverage"]["valid"] is False
    assert len(list((output / "samples").iterdir())) == 4


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing-task", "missing_planned_sample"),
        ("unexpected-task", "unexpected_sample"),
        ("missing-result", "missing_required_artifact"),
        ("missing-telemetry", "missing_required_artifact"),
        ("missing-marker", "complete_marker_integrity_failure"),
        ("provenance", "authorization_provenance_mismatch"),
        ("runtime", "runtime_state_rollback_or_integrity_failure"),
    ],
)
def test_four_task_target_canary_failures_remain_fail_closed(
    tmp_path, monkeypatch, mutation, reason
):
    _plan_value, _root, output, *_authorization = _complete_four_task_target_fixture(
        tmp_path, monkeypatch
    )
    sample = next((output / "samples").iterdir())
    if mutation == "missing-task":
        shutil.rmtree(sample)
    elif mutation == "unexpected-task":
        extra = output / "samples" / "unexpected"
        extra.mkdir()
        runner._atomic_json(extra / "result.json", {})
    elif mutation == "missing-result":
        (sample / "result.json").unlink()
    elif mutation == "missing-telemetry":
        (sample / "telemetry.json").unlink()
    elif mutation == "missing-marker":
        (sample / "COMPLETED.json").unlink()
    elif mutation == "provenance":
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["authorization_provenance"]["task_ids"] = manifest[
            "authorization_provenance"
        ]["task_ids"][:-1]
        runner._atomic_json(output / "manifest.json", manifest)
    else:
        state = json.loads(
            (output / "authorization_runtime.json").read_text(encoding="utf-8")
        )
        state["generation"] += 1
        runner._atomic_json(output / "authorization_runtime.json", state)
    report = runner.validate_artifacts(output)
    assert report["valid"] is False
    assert report["target_canary_acceptance"]["accepted"] is False
    assert reason in report["reason_codes"]
    if mutation == "missing-task":
        assert (
            preflight_cli_main(["validate-artifacts", "--run-dir", str(output)])
            == runner.EXIT_VALIDATION
        )


def test_target_runner_accepts_valid_gate_and_persists_provenance_without_workload(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _fresh_runner_authorization(
        plan, root, monkeypatch
    )
    output = root / environment["output_directory"]
    monkeypatch.setattr(
        runner,
        "capture_environment",
        lambda _root: {
            "schema_version": 2,
            "git_commit": "a" * 40,
            "environment_digest": "fixture",
        },
    )
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_FREE_DISK_BYTES + 1),
    )
    _stop_before_worker(monkeypatch)
    with pytest.raises(_DispatchReached):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["authorization_provenance"]
    assert provenance["authorization_digest"] == authorization["authorization_digest"]
    assert provenance["task_ids"] == authorization["task_ids"]
    assert provenance["normalized_output_directory"] == str(output.resolve())
    assert not (output / "samples").exists()


def test_resume_rejects_replacement_authorization_before_dispatch(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _fresh_runner_authorization(
        plan, root, monkeypatch
    )
    output = root / environment["output_directory"]
    monkeypatch.setattr(
        runner,
        "capture_environment",
        lambda _root: {
            "schema_version": 2,
            "git_commit": "a" * 40,
            "environment_digest": "fixture",
        },
    )
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_FREE_DISK_BYTES + 1),
    )
    _stop_before_worker(monkeypatch)
    with pytest.raises(_DispatchReached):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    replacement = dict(authorization)
    replacement["operator_identity"] = "replacement-operator"
    replacement["authorization_digest"] = canonical_digest(
        replacement, "authorization_digest"
    )
    with pytest.raises(P7C4B2CError, match="authorization_provenance_mismatch"):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=replacement,
        )


def test_target_run_rejects_output_mismatch_and_live_disk_before_mutation(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _fresh_runner_authorization(
        plan, root, monkeypatch
    )
    wrong = root / "artifacts" / "wrong"
    with pytest.raises(P7C4B2CError, match="authorized_output_mismatch"):
        runner.run(
            plan,
            wrong,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert not wrong.exists()
    output = root / environment["output_directory"]
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_FREE_DISK_BYTES - 1),
    )
    with pytest.raises(P7C4B2CError, match="insufficient_live_disk"):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert not output.exists()
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: (_ for _ in ()).throw(OSError("fixture lookup failure")),
    )
    with pytest.raises(P7C4B2CError, match="live_disk_lookup_failed"):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "failure,code",
    [
        ("legacy", "target_resume_provenance_missing"),
        ("disk", "insufficient_live_disk"),
        ("expired", "authorization_expired"),
    ],
)
def test_target_resume_fails_closed_before_dispatch(
    tmp_path, monkeypatch, failure, code
):
    plan, root, output, environment, proposal, authorization = _stopped_target_run(
        tmp_path, monkeypatch
    )
    if failure == "legacy":
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("authorization_provenance")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif failure == "disk":
        monkeypatch.setattr(
            runner.shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(free=MINIMUM_FREE_DISK_BYTES - 1),
        )
    else:
        now = datetime.now(UTC)
        authorization["created_at"] = (now - timedelta(hours=2)).isoformat()
        authorization["expires_at"] = (now - timedelta(hours=1)).isoformat()
        authorization["authorization_digest"] = canonical_digest(
            authorization, "authorization_digest"
        )
    with pytest.raises(P7C4B2CError, match=code):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert not (output / "samples").exists()


def test_resume_accumulated_runtime_uses_conservative_wall_envelope(
    tmp_path, monkeypatch
):
    plan, root, output, environment, proposal, authorization = _stopped_target_run(
        tmp_path, monkeypatch
    )
    state_path = output / "authorization_runtime.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(state["runtime_started_at"])

    class StopOnSubmit:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, *_args, **_kwargs):
            raise _DispatchReached()

    monkeypatch.setattr(
        runner, "ProcessPoolExecutor", lambda *_args, **_kwargs: StopOnSubmit()
    )
    with pytest.raises(_DispatchReached):
        runner._resume_impl(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
            target_authorized=False,
            authorization_plan_digest=None,
            wall_clock=lambda: started + timedelta(seconds=10),
            monotonic_clock=lambda: 100.0,
            disk_usage_provider=lambda _path: SimpleNamespace(
                free=MINIMUM_FREE_DISK_BYTES + 1
            ),
            initial_authorization_validated=True,
        )
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["accumulated_elapsed_seconds"] >= 9.0


@pytest.mark.parametrize(
    "mutation,code",
    [
        (
            lambda task: task.__setitem__("dataset_id", "GMC"),
            "canonical_task_manifest_mismatch",
        ),
        (
            lambda task: task.__setitem__("model_id", "mlp_5"),
            "canonical_task_manifest_mismatch",
        ),
        (
            lambda task: task.__setitem__("seed", 999),
            "canonical_task_manifest_mismatch",
        ),
        (
            lambda task: task.__setitem__("outer_fold", 1 - task["outer_fold"]),
            "canonical_task_manifest_mismatch",
        ),
        (
            lambda task: task.__setitem__("unknown", "field"),
            "canonical_task_manifest_mismatch",
        ),
    ],
)
def test_target_resume_rejects_mutable_manifest_task_payload_before_cleanup(
    tmp_path, monkeypatch, mutation, code
):
    _plan_value, root, output, environment, proposal, authorization = (
        _stopped_target_run(tmp_path, monkeypatch)
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest["expected_tasks"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stale = output / "tmp" / "must-not-be-cleaned"
    stale.mkdir(parents=True)
    with pytest.raises(P7C4B2CError, match=code):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert stale.exists()
    assert not (output / "samples").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["expected_tasks"].pop(),
        lambda manifest: manifest["expected_tasks"].append(
            dict(manifest["expected_tasks"][0])
        ),
        lambda manifest: manifest["expected_tasks"].reverse(),
        lambda manifest: manifest["authorization_provenance"].__setitem__(
            "canonical_task_set_digest", "bad"
        ),
    ],
)
def test_target_resume_rejects_task_set_structure_or_digest_tampering_before_cleanup(
    tmp_path, monkeypatch, mutation
):
    _plan_value, root, output, environment, proposal, authorization = (
        _stopped_target_run(tmp_path, monkeypatch)
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stale = output / "tmp" / "must-not-be-cleaned"
    stale.mkdir(parents=True)
    with pytest.raises(P7C4B2CError):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert stale.exists()
    assert not (output / "samples").exists()


@pytest.mark.parametrize(
    "mutation,code",
    [
        (
            lambda state: state.__setitem__(
                "runtime_started_at",
                (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            ),
            "runtime_state_provenance_mismatch",
        ),
        (
            lambda state: state.__setitem__("accumulated_elapsed_seconds", 0.0),
            "runtime_state_rollback_or_integrity_failure",
        ),
        (
            lambda state: state.__setitem__("accumulated_elapsed_seconds", True),
            "runtime_state_rollback_or_integrity_failure",
        ),
    ],
)
def test_target_resume_rejects_runtime_state_tampering_before_cleanup(
    tmp_path, monkeypatch, mutation, code
):
    _plan_value, root, output, environment, proposal, authorization = (
        _stopped_target_run(tmp_path, monkeypatch)
    )
    state_path = output / "authorization_runtime.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if code == "runtime_state_rollback_or_integrity_failure":
        state["accumulated_elapsed_seconds"] = 10.0
        state["generation"] += 1
        state["state_digest"] = runner._runtime_state_digest(state)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_checkpoint"] = runner._runtime_checkpoint(state)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mutation(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    stale = output / "tmp" / "must-not-be-cleaned"
    stale.mkdir(parents=True)
    with pytest.raises(P7C4B2CError, match=code):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert stale.exists()
    assert not (output / "samples").exists()


def test_target_resume_rejects_trusted_clock_rollback_before_cleanup(
    tmp_path, monkeypatch
):
    _plan_value, root, output, environment, proposal, authorization = (
        _stopped_target_run(tmp_path, monkeypatch)
    )
    state = json.loads(
        (output / "authorization_runtime.json").read_text(encoding="utf-8")
    )
    last = datetime.fromisoformat(state["last_accounted_at"])
    stale = output / "tmp" / "must-not-be-cleaned"
    stale.mkdir(parents=True)
    with pytest.raises(P7C4B2CError, match="runtime_clock_rollback"):
        runner._resume_impl(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
            target_authorized=False,
            authorization_plan_digest=None,
            wall_clock=lambda: last - timedelta(minutes=1),
            monotonic_clock=lambda: 1.0,
            disk_usage_provider=lambda _path: SimpleNamespace(
                free=MINIMUM_FREE_DISK_BYTES + 1
            ),
            initial_authorization_validated=True,
        )
    assert stale.exists()
    assert not (output / "samples").exists()


def test_target_run_submits_exact_authorized_canonical_tasks_without_workload(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _fresh_runner_authorization(
        plan, root, monkeypatch
    )
    output = root / environment["output_directory"]
    submitted = []

    class CaptureExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, fn, task, **kwargs):
            assert fn is runner._execute_task
            submitted.append(task)
            return object()

    monkeypatch.setattr(
        runner, "ProcessPoolExecutor", lambda *_args, **_kwargs: CaptureExecutor()
    )
    monkeypatch.setattr(
        runner,
        "canonical_outer_refit",
        lambda *_args: pytest.fail("canonical workload ran"),
    )
    monkeypatch.setattr(
        runner,
        "capture_environment",
        lambda _root: {
            "schema_version": 2,
            "git_commit": "a" * 40,
            "environment_digest": "fixture",
        },
    )

    def stop_after_all_submissions(_futures):
        if len(submitted) == 4:
            raise _DispatchReached()
        return []

    monkeypatch.setattr(runner, "as_completed", stop_after_all_submissions)
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=MINIMUM_FREE_DISK_BYTES + 1),
    )
    with pytest.raises(_DispatchReached):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    rebuilt = runner.validate_canonical_plan(plan, root)
    expected = [
        next(task for task in rebuilt["tasks"] if task["sample_id"] == task_id)
        for task_id in authorization["task_ids"]
    ]
    assert {task["sample_id"] for task in submitted} == {
        task["sample_id"] for task in expected
    }
    assert all(task in expected for task in submitted)
    assert len(submitted) == 4


def test_expiry_between_tasks_blocks_next_dispatch_without_sleep(tmp_path, monkeypatch):
    plan, root, output, environment, proposal, authorization = _stopped_target_run(
        tmp_path, monkeypatch
    )
    clock = {
        "wall": datetime.now(UTC),
        "mono": 100.0,
        "submits": 0,
    }

    class AdvanceAfterFirstSubmit:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, *_args, **_kwargs):
            clock["submits"] += 1
            clock["wall"] += timedelta(hours=2)
            clock["mono"] += 7200.0
            return object()

    monkeypatch.setattr(
        runner,
        "ProcessPoolExecutor",
        lambda *_args, **_kwargs: AdvanceAfterFirstSubmit(),
    )
    with pytest.raises(P7C4B2CError, match="authorization_expired"):
        runner._resume_impl(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
            target_authorized=False,
            authorization_plan_digest=None,
            wall_clock=lambda: clock["wall"],
            monotonic_clock=lambda: clock["mono"],
            disk_usage_provider=lambda _path: SimpleNamespace(
                free=MINIMUM_FREE_DISK_BYTES + 1
            ),
            initial_authorization_validated=True,
        )
    assert clock["submits"] == 1
    assert not (output / "samples").exists()


def _redigest_plan(plan):
    for task in plan.get("tasks", []):
        if isinstance(task, dict) and "sample_id" in task:
            task["sample_id"] = runner.sha256_canonical(
                {key: value for key, value in task.items() if key != "sample_id"}
            )
    plan["plan_digest"] = canonical_digest(plan, "plan_digest")


def _mutate_candidate(plan, mutation):
    mutation(plan["tasks"][0]["candidate"])


def _append_unique_task(plan):
    task = json.loads(json.dumps(plan["tasks"][0]))
    task["seed"] += 100
    plan["tasks"].append(task)


def _replace_candidate_everywhere(plan):
    candidate_id = plan["tasks"][0]["candidate_id"]
    for task in plan["tasks"]:
        if task["candidate_id"] == candidate_id:
            task["candidate"]["hidden_units"] = [999]


def _duplicate_candidate_id(plan):
    plan["tasks"][0]["candidate_id"] = next(
        task["candidate_id"]
        for task in plan["tasks"]
        if task["candidate_id"] != plan["tasks"][0]["candidate_id"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["tasks"][0]["candidate"].__setitem__(
            "hidden_units", [999, 999, 999]
        ),
        _replace_candidate_everywhere,
        lambda plan: plan["tasks"][0].__setitem__("candidate_id", "foreign-candidate"),
        _duplicate_candidate_id,
        lambda plan: _mutate_candidate(plan, lambda value: value.pop("dropout")),
        lambda plan: _mutate_candidate(
            plan, lambda value: value.__setitem__("unknown_hyperparameter", 1.0)
        ),
        lambda plan: _mutate_candidate(
            plan, lambda value: value.__setitem__("hidden_units", "not-a-list")
        ),
        lambda plan: _mutate_candidate(
            plan, lambda value: value.__setitem__("dropout", 1.5)
        ),
        lambda plan: _mutate_candidate(
            plan,
            lambda value: value.__setitem__(
                "batch_normalization", not value["batch_normalization"]
            ),
        ),
        lambda plan: plan["tasks"][0].__setitem__("dataset_id", "GMC"),
        lambda plan: plan["tasks"][0].__setitem__("model_id", "mlp_5"),
        lambda plan: plan["tasks"][0].__setitem__("candidate_proxy", "high_cost_proxy"),
        lambda plan: plan["tasks"][0].__setitem__("seed", 999),
        lambda plan: plan["tasks"][0].__setitem__("repetition", 1),
        lambda plan: plan["tasks"][0].__setitem__("outer_fold", 1),
        lambda plan: plan["execution"].__setitem__(
            "target_authorization", "foreign-policy"
        ),
        _append_unique_task,
        lambda plan: plan["tasks"].pop(),
        lambda plan: plan["tasks"].append(dict(plan["tasks"][0])),
        lambda plan: plan["tasks"].reverse(),
    ],
    ids=(
        "hidden-units",
        "locked-candidate-space",
        "candidate-id",
        "duplicate-candidate-id",
        "candidate-missing-key",
        "candidate-extra-key",
        "candidate-wrong-type",
        "candidate-range",
        "refit-config",
        "dataset",
        "model",
        "proxy",
        "seed",
        "repetition",
        "fold",
        "plan-execution",
        "task-added",
        "task-removed",
        "task-duplicate",
        "task-reordered",
    ),
)
def test_redigested_persisted_plan_tampering_fails_before_every_side_effect(
    tmp_path, monkeypatch, mutation
):
    _plan_value, root, output, environment, proposal, authorization = (
        _stopped_target_run(tmp_path, monkeypatch)
    )
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mutation(plan)
    _redigest_plan(plan)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    stale = output / "tmp" / "must-not-be-cleaned"
    stale.mkdir(parents=True)
    runtime_path = output / "authorization_runtime.json"
    runtime_before = runtime_path.read_bytes()
    manifest_before = (output / "manifest.json").read_bytes()
    effects = {"executor": 0, "submit": 0, "workload": 0}

    def forbidden_executor(*_args, **_kwargs):
        effects["executor"] += 1
        raise AssertionError("executor constructed")

    def forbidden_workload(*_args, **_kwargs):
        effects["workload"] += 1
        raise AssertionError("canonical workload called")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", forbidden_executor)
    monkeypatch.setattr(runner, "canonical_outer_refit", forbidden_workload)
    with pytest.raises(P7C4B2CError, match="plan_"):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert stale.exists()
    assert not (output / "samples").exists()
    assert runtime_path.read_bytes() == runtime_before
    assert (output / "manifest.json").read_bytes() == manifest_before
    assert effects == {"executor": 0, "submit": 0, "workload": 0}


def test_target_resume_submits_only_pending_rebuilt_canonical_tasks(
    tmp_path, monkeypatch
):
    plan, root, output, environment, proposal, authorization = _stopped_target_run(
        tmp_path, monkeypatch
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    persisted_environment = json.loads(
        (output / "environment.json").read_text(encoding="utf-8")
    )
    completed_task = next(
        task
        for task in plan["tasks"]
        if task["sample_id"] == authorization["task_ids"][0]
    )
    record = {
        **completed_task,
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "attempt": 1,
        "attempt_id": "fixture-attempt",
        "execution_class": "target_preflight",
        "git_commit": persisted_environment["git_commit"],
        "plan_digest": plan["plan_digest"],
        "preprocessing_identity": "fixture",
        "input_identity": {},
        "input_hash": runner.sha256_canonical({}),
        "started_utc": "2026-08-10T00:00:00Z",
        "completed_utc": "2026-08-10T00:00:00Z",
        "started_monotonic": 0.0,
        "completed_monotonic": 0.0,
        "status": "completed",
        "projection_eligible": False,
        "limitations": [],
        "result": {},
        **{field: 0.0 for field in TIMING_FIELDS},
    }
    sample_dir = output / "samples" / completed_task["sample_id"]
    sample_dir.mkdir(parents=True)
    (sample_dir / "result.json").write_text(json.dumps(record), encoding="utf-8")
    (sample_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "record_digest": runner.sha256_canonical(record),
                "attempt_id": record["attempt_id"],
            }
        ),
        encoding="utf-8",
    )
    submitted = []

    class CaptureExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, fn, task, **_kwargs):
            assert fn is runner._execute_task
            assert (output / "authorization_runtime.json").exists()
            assert (output / "manifest.json").exists()
            submitted.append(task)
            return object()

    monkeypatch.setattr(
        runner, "ProcessPoolExecutor", lambda *_args, **_kwargs: CaptureExecutor()
    )
    monkeypatch.setattr(
        runner,
        "canonical_outer_refit",
        lambda *_args: pytest.fail("canonical workload ran"),
    )

    def stop_after_pending_submissions(_futures):
        if len(submitted) == 3:
            raise _DispatchReached()
        return []

    monkeypatch.setattr(runner, "as_completed", stop_after_pending_submissions)
    with pytest.raises(_DispatchReached):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    rebuilt = runner.validate_canonical_plan(plan, root)
    expected = [
        task
        for task in rebuilt["tasks"]
        if task["sample_id"] in authorization["task_ids"]
        and task["sample_id"] != completed_task["sample_id"]
    ]
    assert len(submitted) == 3
    assert {task["sample_id"] for task in submitted} == {
        task["sample_id"] for task in expected
    }
    assert all(task in expected for task in submitted)


def _mutate_protocol(root: Path) -> None:
    path = root / "configs/protocols/protocol_a.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("strategy: mean", "strategy: median"),
        encoding="utf-8",
    )


def _mutate_dataset_semantics(root: Path) -> None:
    path = root / "data/datasets.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "mapping_to_binary: {'0': 0, '1': 1}",
            "mapping_to_binary: {'0': 1, '1': 0}",
            1,
        ),
        encoding="utf-8",
    )


def _mutate_selected_checksum(root: Path) -> None:
    path = root / "data/checksums-sha256.csv"
    text = path.read_text(encoding="utf-8")
    first_hash = hashlib.sha256(b"ac fixture\n").hexdigest().upper()
    path.write_text(text.replace(first_hash, "0" * 64), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    [_mutate_protocol, _mutate_dataset_semantics, _mutate_selected_checksum],
    ids=["protocol-a", "dataset-semantics", "selected-checksum"],
)
def test_locked_runtime_mutation_fails_before_resume_side_effects(
    tmp_path, monkeypatch, mutation
):
    _plan_value, root, output, environment, proposal, authorization = (
        _stopped_target_run(tmp_path, monkeypatch)
    )
    stale = output / "tmp" / "must-not-be-cleaned"
    stale.mkdir(parents=True)
    runtime_path = output / "authorization_runtime.json"
    manifest_path = output / "manifest.json"
    runtime_before = runtime_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    effects = {"executor": 0, "submit": 0, "workload": 0}

    mutation(root)

    def forbidden_executor(*_args, **_kwargs):
        effects["executor"] += 1
        raise AssertionError("executor constructed")

    def forbidden_workload(*_args, **_kwargs):
        effects["workload"] += 1
        raise AssertionError("workload invoked")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", forbidden_executor)
    monkeypatch.setattr(runner, "canonical_outer_refit", forbidden_workload)
    with pytest.raises(P7C4B2CError, match="locked_runtime_input_mismatch"):
        runner.resume(
            output,
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert stale.exists()
    assert runtime_path.read_bytes() == runtime_before
    assert manifest_path.read_bytes() == manifest_before
    assert effects == {"executor": 0, "submit": 0, "workload": 0}


def test_locked_runtime_mutation_fails_before_run_output_or_executor(
    tmp_path, monkeypatch
):
    plan, root = _plan(), _repo(tmp_path)
    environment, proposal, authorization = _fresh_runner_authorization(
        plan, root, monkeypatch
    )
    output = root / environment["output_directory"]
    effects = {"executor": 0, "workload": 0}
    _mutate_dataset_semantics(root)

    def forbidden_executor(*_args, **_kwargs):
        effects["executor"] += 1
        raise AssertionError("executor constructed")

    def forbidden_workload(*_args, **_kwargs):
        effects["workload"] += 1
        raise AssertionError("workload invoked")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", forbidden_executor)
    monkeypatch.setattr(runner, "canonical_outer_refit", forbidden_workload)
    with pytest.raises(P7C4B2CError, match="locked_runtime_input_mismatch"):
        runner.run(
            plan,
            output,
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            repo_root=root,
            target_environment=environment,
            authorization_proposal=proposal,
            effective_authorization=authorization,
        )
    assert not output.exists()
    assert effects == {"executor": 0, "workload": 0}


@pytest.mark.parametrize("failure", ["missing", "malformed", "coercible-registry"])
def test_locked_runtime_missing_or_malformed_fails_closed(tmp_path, failure):
    root = _repo(tmp_path)
    path = root / "configs/protocols/protocol_a.yaml"
    if failure == "missing":
        path.unlink()
    elif failure == "malformed":
        path.write_text("protocol: [not-a-mapping]\n", encoding="utf-8")
    else:
        registry_path = root / "data/datasets.yaml"
        registry_path.write_text(
            registry_path.read_text(encoding="utf-8").replace(
                "mapping_to_binary: {'0': 0, '1': 1}",
                "mapping_to_binary: {'0': '0', '1': 1}",
                1,
            ),
            encoding="utf-8",
        )
    with pytest.raises(Exception, match="invalid"):
        load_locked_runtime_inputs(root)


def test_irrelevant_registry_metadata_does_not_change_semantic_digest(tmp_path):
    root = _repo(tmp_path)
    before = load_locked_runtime_inputs(root).digest
    path = root / "data/datasets.yaml"
    path.write_text(
        path.read_text(encoding="utf-8") + "documentation_only: fixture note\n",
        encoding="utf-8",
    )
    assert load_locked_runtime_inputs(root).digest == before


def test_worker_rechecks_after_parent_preflight_before_workload(tmp_path, monkeypatch):
    plan, root, output, _environment_value, _proposal, authorization = (
        _stopped_target_run(tmp_path, monkeypatch)
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    persisted_environment = json.loads(
        (output / "environment.json").read_text(encoding="utf-8")
    )
    task = next(
        item
        for item in plan["tasks"]
        if item["sample_id"] == authorization["task_ids"][0]
    )
    temp_before = sorted(path.as_posix() for path in (output / "tmp").rglob("*"))
    effects = {"workload": 0}
    _mutate_protocol(root)

    def forbidden_workload(*_args, **_kwargs):
        effects["workload"] += 1
        raise AssertionError("workload invoked")

    monkeypatch.setattr(runner, "canonical_outer_refit", forbidden_workload)
    with pytest.raises(P7C4B2CError, match="locked_runtime_input_mismatch"):
        runner._execute_task(
            task,
            manifest=manifest,
            environment=persisted_environment,
            run_dir=output,
            repo_root=root,
            attempt=1,
            authorization_deadline_monotonic=time.perf_counter() + 60,
        )
    assert effects == {"workload": 0}
    assert (
        sorted(path.as_posix() for path in (output / "tmp").rglob("*")) == temp_before
    )


@pytest.mark.parametrize(
    "mutation,duplicate_key",
    [
        (
            lambda text: text + "protocol:\n  name: protocol_a\n",
            "protocol",
        ),
        (
            lambda text: text.replace(
                "  version: p3b-v1\n",
                "  version: p3b-v1\n  version: duplicate\n",
                1,
            ),
            "version",
        ),
    ],
    ids=["top-level", "nested"],
)
def test_protocol_yaml_duplicate_keys_fail_closed(tmp_path, mutation, duplicate_key):
    root = _repo(tmp_path)
    path = root / "configs/protocols/protocol_a.yaml"
    path.write_text(mutation(path.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(PreprocessingError) as raised:
        load_protocol_a_config(repo_root=root)

    assert f"duplicate_yaml_mapping_key:'{duplicate_key}'" in str(raised.value)


@pytest.mark.parametrize(
    "mutation,duplicate_key",
    [
        (
            lambda text: text.replace(
                "datasets:\n  ac:\n", "datasets:\n  ac: {}\n  ac:\n", 1
            ),
            "ac",
        ),
        (
            lambda text: text.replace(
                "reader: {type: csv}", "reader: {type: csv, type: delimited}", 1
            ),
            "type",
        ),
    ],
    ids=["selected-dataset", "nested-reader"],
)
def test_selected_dataset_yaml_duplicate_keys_fail_closed(
    tmp_path, mutation, duplicate_key
):
    root = _repo(tmp_path)
    path = root / "data/datasets.yaml"
    path.write_text(mutation(path.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(LockedRuntimeInputError) as raised:
        load_locked_runtime_inputs(root)

    assert raised.value.args == ("dataset_registry_invalid",)
    assert raised.value.__cause__ is not None
    assert f"duplicate_yaml_mapping_key:'{duplicate_key}'" in str(
        raised.value.__cause__
    )


@pytest.mark.parametrize("version", ["123", "[p3b-v1]", "null"])
def test_protocol_version_requires_nonempty_string(tmp_path, version):
    root = _repo(tmp_path)
    path = root / "configs/protocols/protocol_a.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("p3b-v1", version, 1),
        encoding="utf-8",
    )

    with pytest.raises(PreprocessingError, match="version must be a non-empty string"):
        load_protocol_a_config(repo_root=root)


@pytest.mark.parametrize("value", ['"0.5"', "true", ".nan", ".inf"])
def test_protocol_numeric_fields_reject_coercible_or_nonfinite_values(tmp_path, value):
    root = _repo(tmp_path)
    path = root / "configs/protocols/protocol_a.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "smoothing: 0.5", f"smoothing: {value}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(PreprocessingError, match="smoothing must be > 0"):
        load_protocol_a_config(repo_root=root)


@pytest.mark.parametrize("header", ["0", "1", '"false"'])
def test_selected_reader_header_rejects_non_boolean_values(tmp_path, header):
    root = _repo(tmp_path)
    path = root / "data/datasets.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "reader: {type: csv}",
            f"reader: {{type: delimited, header: {header}}}",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(LockedRuntimeInputError, match="reader.header"):
        load_locked_runtime_inputs(root)


def test_valid_locked_runtime_yaml_still_loads(tmp_path):
    locked = load_locked_runtime_inputs(_repo(tmp_path))

    assert locked.protocol_config.protocol_version == "p3b-v1"
    assert set(locked.registry) == {"ac", "gmc"}
