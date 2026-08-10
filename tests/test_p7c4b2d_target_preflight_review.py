from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from creditrep.experiments import p7c4b2d_cli as cli
from creditrep.experiments.p7c4b2d_cli import EXIT_REVIEW_BLOCKED, main as cli_main
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2c import build_plan
from creditrep.protocols.p7c4b2c import canonical_digest
from creditrep.protocols.p7c4b2d import (
    DISK_POLICY,
    MINIMUM_FREE_DISK_BYTES,
    P7C4B2DError,
    collect_target_environment,
    dependency_lock_fingerprint,
    decision_package,
    environment_digest,
    merge_operator_metadata,
    render_authorization_proposal,
    select_canary,
    validate_authorization_proposal,
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
    for dataset, payload in {"ac": b"ac fixture\n", "gmc": b"gmc fixture\n"}.items():
        folder = tmp_path / "data" / "raw" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{dataset}.csv").write_bytes(payload)
    registry = """datasets:
  ac:
    id: ac
    active_file: data/raw/ac/ac.csv
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
    for dataset in ("ac", "gmc"):
        payload = (tmp_path / "data" / "raw" / dataset / f"{dataset}.csv").read_bytes()
        checks.append(
            f'"data/raw/{dataset}/{dataset}.csv","SHA256","{hashlib.sha256(payload).hexdigest().upper()}"'
        )
    (tmp_path / "data" / "checksums-sha256.csv").write_text(
        "\n".join(checks) + "\n", encoding="utf-8"
    )
    (tmp_path / "artifacts").mkdir()
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
        "provider": "fixture",
        "region": "fixture",
        "instance_id": "fixture",
        "os": "fixture",
        "python_version": "3.11",
        "cpu_model": "fixture",
        "vcpu_count": 2,
        "ram_bytes": 8,
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
        "output_directory": "artifacts/p7c4b2d-fixture",
        "hourly_price": 2.5,
        "currency": "USD",
        "price_source": "fixture",
        "price_observed_at": "2026-08-10T00:00:00Z",
        "maximum_runtime_hours": 10.0,
        "maximum_monetary_budget": 25.0,
        "evidence_observed_at": "2026-08-10T00:00:00Z",
    }
    value["environment_digest"] = environment_digest(value)
    return value


def _pass_probe():
    return {"probe": "fixture", "status": "pass", "timeout_seconds": 1}


def _operator_metadata():
    return {
        "provider": "fixture-cloud",
        "region": "fixture-region",
        "instance_id": "fixture-instance",
        "disk_type": "ssd",
        "network_topology": "single_vm",
        "vm_count": 1,
        "hourly_price": 2.5,
        "currency": "USD",
        "price_source": "fixture-invoice",
        "price_observed_at": "2026-08-10T00:00:00Z",
        "maximum_runtime_hours": 10.0,
        "maximum_monetary_budget": 25.0,
    }


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
