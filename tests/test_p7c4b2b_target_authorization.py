from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytest

from creditrep.protocols.p7c4b2b import (
    DATASET_FINGERPRINTS,
    SCIENTIFIC_DIGEST,
    machine_profile_digest,
    PreflightError,
)
from creditrep.experiments.p7c4b2b_preflight import execution_guard, load_default_plan
from creditrep.experiments.p7c4b2b_cli import main as cli_main
from creditrep.protocols.p7c4b2b_authorization import (
    APPROVAL_PHRASE,
    authorization_digest,
    create_effective_authorization,
    environment_digest,
    proposal_digest,
    render_authorization_proposal,
    render_target_environment,
    validate_authorization_proposal,
    validate_effective_authorization,
    validate_target_environment,
)

NOW = "2026-08-14T00:00:00Z"
EXPIRY = "2099-08-14T00:00:00Z"


def _head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _profile(root: Path, *, identity: str = "one") -> dict:
    value = {
        "schema_version": 1,
        "machine_role": "intended_single_vm_target",
        "machine_id": identity * 16,
        "cloud_provider": "fixture-provider",
        "instance_type": f"fixture-{identity}",
        "os": "fixture-linux",
        "cpu_model": "fixture-cpu",
        "physical_cores": 8,
        "logical_cores": 16,
        "ram_total_bytes": 32 * 1024**3,
        "system_used_ram_bytes": 1024**3,
        "swap_total_bytes": 0,
        "swap_used_bytes": 0,
        "disk_free_bytes": 64 * 1024**3,
        "python_executable": ".venv/bin/python",
        "python_version": "3.11",
        "dependency_fingerprint": "d" * 64,
        "git_commit": _head(root),
        "scientific_manifest_digest": SCIENTIFIC_DIGEST,
        "dataset_fingerprints": DATASET_FINGERPRINTS,
        "worker_limit": 2,
        "threads_per_worker": 2,
        "virtualization_power": "unknown",
        "utc_captured": NOW,
    }
    value["profile_digest"] = machine_profile_digest(value)
    return value


def _chain(root: Path, tmp_path: Path, mode: str, *, identity: str = "one"):
    plan = load_default_plan(root)
    profile = _profile(root, identity=identity)
    output = tmp_path / f"inner-{mode}-{identity}"
    environment = render_target_environment(
        plan, profile, mode=mode, output_directory=output, captured_at=NOW
    )
    proposal = render_authorization_proposal(
        environment,
        plan,
        profile,
        run_id=output.name,
        created_at=NOW,
    )
    authorization = create_effective_authorization(
        proposal,
        environment,
        plan,
        profile,
        operator_identity="reviewer@example.invalid",
        operator_approval=APPROVAL_PHRASE,
        created_at=NOW,
        expires_at=EXPIRY,
    )
    return plan, profile, output, environment, proposal, authorization


@pytest.mark.parametrize("mode,workers", [("cpu_parallel_1", 1), ("cpu_parallel_2", 2)])
def test_valid_typed_chain_has_exact_mode_scope(tmp_path, mode, workers):
    root = Path.cwd().resolve()
    plan, profile, output, environment, proposal, authorization = _chain(
        root, tmp_path, mode, identity=mode[-1]
    )
    assert environment["task_count"] == len(environment["task_ids"]) == 54
    assert environment["worker_count"] == workers
    assert environment["normalized_output_directory"] == str(output.resolve())
    assert proposal["authorization_effective"] is False
    assert authorization["authorization_effective"] is True
    assert validate_target_environment(environment, plan, profile)["valid"]
    assert validate_authorization_proposal(proposal, environment, plan, profile)[
        "valid"
    ]
    assert validate_effective_authorization(
        authorization,
        proposal,
        environment,
        plan,
        profile,
        now=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )["valid"]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda value: value.pop("task_count"), "environment_schema_invalid"),
        (
            lambda value: value.__setitem__("unknown", True),
            "environment_schema_invalid",
        ),
        (
            lambda value: value.__setitem__("source_git_commit", "A" * 40),
            "source_git_commit_invalid",
        ),
        (
            lambda value: value.__setitem__("source_git_commit", "a" * 39),
            "source_git_commit_invalid",
        ),
        (
            lambda value: value.__setitem__("scientific_manifest_digest", "0" * 64),
            "scientific_manifest_digest_mismatch",
        ),
        (
            lambda value: value.__setitem__("plan_digest", "0" * 64),
            "plan_digest_mismatch",
        ),
        (
            lambda value: value.__setitem__("mode", "cpu_parallel_2"),
            "authorization_task_scope_mismatch",
        ),
        (lambda value: value["task_ids"].pop(), "authorization_task_scope_mismatch"),
        (
            lambda value: value["task_ids"].append(value["task_ids"][0]),
            "authorization_task_scope_mismatch",
        ),
        (
            lambda value: value["task_ids"].reverse(),
            "authorization_task_scope_mismatch",
        ),
        (
            lambda value: value.__setitem__("machine_profile_digest", "0" * 64),
            "machine_profile_digest_mismatch",
        ),
        (
            lambda value: value.__setitem__("normalized_output_directory", "relative"),
            "output_identity_invalid",
        ),
        (
            lambda value: value.__setitem__("environment_digest", "0" * 64),
            "environment_digest_mismatch",
        ),
    ],
)
def test_environment_mutations_fail_closed(tmp_path, mutation, reason):
    root = Path.cwd().resolve()
    plan, profile, _output, environment, _proposal, _authorization = _chain(
        root, tmp_path, "cpu_parallel_1"
    )
    mutation(environment)
    report = validate_target_environment(environment, plan, profile)
    assert report["valid"] is False
    assert reason in report["reason_codes"]


def test_redigested_cross_artifact_mutations_and_wrong_modes_fail(tmp_path):
    root = Path.cwd().resolve()
    plan, profile, _output, environment, proposal, authorization = _chain(
        root, tmp_path, "cpu_parallel_1"
    )
    changed_environment = deepcopy(environment)
    changed_environment["worker_count"] = 2
    changed_environment["environment_digest"] = environment_digest(changed_environment)
    assert not validate_target_environment(changed_environment, plan, profile)["valid"]

    changed_proposal = deepcopy(proposal)
    changed_proposal["mode"] = "cpu_parallel_2"
    changed_proposal["proposal_digest"] = proposal_digest(changed_proposal)
    assert not validate_authorization_proposal(
        changed_proposal, environment, plan, profile
    )["valid"]

    changed_authorization = deepcopy(authorization)
    changed_authorization["normalized_output_directory"] += "-replacement"
    changed_authorization["authorization_digest"] = authorization_digest(
        changed_authorization
    )
    assert not validate_effective_authorization(
        changed_authorization, proposal, environment, plan, profile
    )["valid"]


@pytest.mark.parametrize(
    "operator,approval,created,expires,reason",
    [
        ("", APPROVAL_PHRASE, NOW, EXPIRY, "operator_identity_missing"),
        (
            "operator",
            "APPROVE_P7C4B2_TARGET_PROJECTION_PREFLIGHT",
            NOW,
            EXPIRY,
            "operator_approval_invalid",
        ),
        (
            "operator",
            "APPROVE_P7C4B2_TARGET_CANARY",
            NOW,
            EXPIRY,
            "operator_approval_invalid",
        ),
        (
            "operator",
            "APPROVE_CANONICAL_SCIENTIFIC_EXECUTION",
            NOW,
            EXPIRY,
            "operator_approval_invalid",
        ),
        ("operator", APPROVAL_PHRASE, EXPIRY, NOW, "authorization_expiry_invalid"),
    ],
)
def test_effective_authorization_ceremony_is_strict(
    tmp_path, operator, approval, created, expires, reason
):
    root = Path.cwd().resolve()
    plan, profile, _output, environment, proposal, _authorization = _chain(
        root, tmp_path, "cpu_parallel_1"
    )
    with pytest.raises(PreflightError, match=reason):
        create_effective_authorization(
            proposal,
            environment,
            plan,
            profile,
            operator_identity=operator,
            operator_approval=approval,
            created_at=created,
            expires_at=expires,
        )


def test_boolean_only_target_execution_is_rejected(tmp_path):
    root = Path.cwd().resolve()
    plan = load_default_plan(root)
    profile = _profile(root)
    output = root / "artifacts/p7c4b2b-compute-preflight/boolean-only-never-created"
    report = execution_guard(
        plan=plan,
        profile=profile,
        output_dir=output,
        mode="cpu_parallel_1",
        fixture=False,
        bounded_authorized=True,
        repo_root=root,
    )
    assert report["authorized"] is False
    assert {
        "typed_target_authorization_missing",
        "boolean_target_authorization_rejected",
    } <= set(report["reason_codes"])


def test_p1_chain_cannot_authorize_p2(tmp_path):
    root = Path.cwd().resolve()
    plan, profile, _output, environment, proposal, authorization = _chain(
        root, tmp_path, "cpu_parallel_1"
    )
    report = execution_guard(
        plan=plan,
        profile=profile,
        output_dir=Path(environment["normalized_output_directory"]),
        mode="cpu_parallel_2",
        fixture=False,
        bounded_authorized=False,
        repo_root=root,
        target_environment=environment,
        authorization_proposal=proposal,
        effective_authorization=authorization,
    )
    assert report["authorized"] is False
    assert "authorization_mode_mismatch" in report["reason_codes"]


@pytest.mark.parametrize("mode", ["cpu_parallel_1", "cpu_parallel_2"])
def test_valid_typed_chain_authorizes_only_its_fresh_target_identity(tmp_path, mode):
    root = Path.cwd().resolve()
    output = (
        root / "artifacts/p7c4b2b-compute-preflight" / f"typed-{mode}-never-created"
    )
    plan = load_default_plan(root)
    profile = _profile(root, identity=mode[-1])
    environment = render_target_environment(
        plan, profile, mode=mode, output_directory=output, captured_at=NOW
    )
    proposal = render_authorization_proposal(
        environment, plan, profile, run_id=output.name, created_at=NOW
    )
    authorization = create_effective_authorization(
        proposal,
        environment,
        plan,
        profile,
        operator_identity="operator",
        operator_approval=APPROVAL_PHRASE,
        created_at=NOW,
        expires_at=EXPIRY,
    )
    report = execution_guard(
        plan=plan,
        profile=profile,
        output_dir=output,
        mode=mode,
        fixture=False,
        bounded_authorized=False,
        repo_root=root,
        target_environment=environment,
        authorization_proposal=proposal,
        effective_authorization=authorization,
    )
    assert report == {"authorized": True, "reason_codes": []}


def test_typed_cli_validation_commands_emit_machine_json(tmp_path, capsys):
    root = Path.cwd().resolve()
    plan, profile, _output, environment, proposal, authorization = _chain(
        root, tmp_path, "cpu_parallel_1"
    )
    profile_path = tmp_path / "profile.json"
    environment_path = tmp_path / "environment.json"
    proposal_path = tmp_path / "proposal.json"
    authorization_path = tmp_path / "authorization.json"
    for path, value in (
        (profile_path, profile),
        (environment_path, environment),
        (proposal_path, proposal),
        (authorization_path, authorization),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")

    assert (
        cli_main(
            [
                "review-target-plan",
                "--profile",
                str(profile_path),
                "--target-environment",
                str(environment_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["authorization_effective"] is False
    assert (
        cli_main(
            [
                "validate-authorization-proposal",
                "--profile",
                str(profile_path),
                "--target-environment",
                str(environment_path),
                "--authorization-proposal",
                str(proposal_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert (
        cli_main(
            [
                "validate-effective-authorization",
                "--profile",
                str(profile_path),
                "--target-environment",
                str(environment_path),
                "--authorization-proposal",
                str(proposal_path),
                "--effective-authorization",
                str(authorization_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert plan["fit_budget"]["no_retry_total"] == 108
