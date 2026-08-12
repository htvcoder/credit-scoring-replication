from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest

from creditrep.experiments import p7c4b2d_cli as cli
from creditrep.experiments import p7c4b2c_preflight as runner
from creditrep.experiments.p7c4b2d_cli import EXIT_REVIEW_BLOCKED, main as cli_main
from creditrep.locked_runtime_inputs import (
    LockedRuntimeInputError,
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
    TARGET_CANARY_APPROVAL,
    _create_effective_authorization,
    _static_authorization_cost_upper_bound,
    collect_target_environment,
    create_effective_authorization,
    dependency_lock_fingerprint,
    decision_package,
    environment_digest,
    merge_operator_metadata,
    normalize_target_output,
    render_authorization_proposal,
    select_canary,
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
        "locked_runtime_inputs_digest": load_locked_runtime_inputs(root).digest,
        "output_directory": "artifacts/p7c4b2d-fixture",
        "hourly_price": "2.5",
        "currency": "USD",
        "price_source": "fixture",
        "price_observed_at": "2026-08-10T00:00:00Z",
        "maximum_runtime_hours": 10.0,
        "maximum_monetary_budget": "25.0",
        "evidence_observed_at": "2026-08-10T00:00:00Z",
        "process_spawn_probe": _pass_probe(),
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
