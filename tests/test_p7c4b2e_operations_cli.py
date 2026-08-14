from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from creditrep.experiments import p7c4b2e_operations_cli as operations
from creditrep.experiments.p7c4b2e_operations_cli import (
    OperationsError,
    UNIT_SNAPSHOT_FIELDS,
    create_launch_record,
    create_submission_claim,
    create_submission_receipt,
    resume_precheck,
)


RUNBOOK = Path(__file__).parents[1] / "docs" / "P7C4B2B_SINGLE_VM_PREFLIGHT_RUNBOOK.md"


def _logical_shell_commands(text):
    command = ""
    for line in text.splitlines():
        stripped = line.strip()
        command += stripped[:-1] + " " if stripped.endswith("\\") else stripped
        if not stripped.endswith("\\"):
            if command:
                yield command
            command = ""


def _runbook_argv(command):
    jq_command = command[command.index("$(") + 2 : -1]
    tokens = shlex.split(jq_command)
    assert tokens[:5] == ["jq", "-cn", "--args", "$ARGS.positional", "--"]
    return tokens[5:]


def test_b2b_runbook_jq_argv_matches_exact_submitted_commands():
    commands = list(_logical_shell_commands(RUNBOOK.read_text(encoding="utf-8")))
    constructions = {
        command.split("=", 1)[0]: _runbook_argv(command)
        for command in commands
        if "jq -cn --args" in command
    }
    assert set(constructions) == {"ARGV_P1", "RESUME_ARGV_P1"}

    expected_run = [
        "$PYTHON",
        "-m",
        "creditrep.experiments.p7c4b2b_cli",
        "run",
        "--mode",
        "cpu_parallel_1",
        "--profile",
        "$PROFILE_P1",
        "--target-machine-asserted",
        "--target-environment",
        "$ENV_P1",
        "--authorization-proposal",
        "$PROPOSAL_P1",
        "--effective-authorization",
        "$AUTH_P1",
        "--output-dir",
        "$OUT_P1",
    ]
    expected_resume = [*expected_run[:3], "resume", *expected_run[4:]]
    assert constructions == {
        "ARGV_P1": expected_run,
        "RESUME_ARGV_P1": expected_resume,
    }

    submitted = {}
    for command in commands:
        if command.startswith("systemd-run ") and "p7c4b2b_cli" in command:
            tokens = shlex.split(command)
            argv = tokens[tokens.index("$PYTHON") :]
            submitted[argv[3]] = argv
    assert submitted == {"run": expected_run, "resume": expected_resume}
    assert not {"--fixture", "--max-tasks", "--timeout-seconds"}.intersection(
        token for argv in constructions.values() for token in argv
    )


def _input(path, value):
    path.write_text(value, encoding="utf-8")
    return path


def _launch(tmp_path):
    auth = _input(tmp_path / "authorization.json", "{}")
    environment = _input(tmp_path / "environment.json", "{}")
    proposal = _input(tmp_path / "proposal.json", "{}")
    record = tmp_path / "launch.json"
    return record, create_launch_record(
        record_path=record,
        git_commit="a" * 40,
        operator_identity="operator",
        authorization_path=auth,
        environment_path=environment,
        proposal_path=proposal,
        unit="p7c4b2d-target",
        argv=["python", "-m", "creditrep"],
        working_directory="/srv/credit-scoring-replication",
        python_executable="/srv/credit-scoring-replication/.venv/bin/python",
        output_directory="/srv/credit-scoring-replication/artifacts/p7c4b2d-target",
        log_path="/secure/p7c4b2d/target.log",
    )


def test_launch_record_is_immutable_and_hashes_inputs(tmp_path):
    record, value = _launch(tmp_path)
    stored = json.loads(record.read_text(encoding="utf-8"))
    assert stored == value
    assert stored["submission_state"] == "prepared_not_submitted"
    assert stored["authorization"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    try:
        _launch(tmp_path)
    except Exception as exc:
        assert str(exc) == "operational_evidence_collision"
    else:
        raise AssertionError("launch record was overwritten")


def test_submission_receipt_requires_complete_snapshot_and_is_immutable(tmp_path):
    record, _value = _launch(tmp_path)
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    receipt = tmp_path / "receipt.json"
    value = create_submission_receipt(
        receipt_path=receipt,
        launch_record_path=record,
        unit_snapshot=snapshot,
        systemd_run_exit_code=0,
    )
    assert (
        value["launch_record_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    )
    assert value["unit_snapshot"] == snapshot
    try:
        create_submission_receipt(
            receipt_path=receipt,
            launch_record_path=record,
            unit_snapshot=snapshot,
            systemd_run_exit_code=0,
        )
    except Exception as exc:
        assert str(exc) == "operational_evidence_collision"
    else:
        raise AssertionError("receipt was overwritten")


def test_canary_evidence_contract_is_exactly_backward_compatible(tmp_path):
    record, launch = _launch(tmp_path)
    assert launch["artifact_type"] == "p7c4b2e_target_canary_launch_record"
    assert launch["schema_version"] == 1
    assert set(launch) == {
        "schema_version",
        "artifact_type",
        "source_git_commit",
        "created_at",
        "operator_identity",
        "authorization",
        "environment",
        "proposal",
        "systemd_unit",
        "argv",
        "working_directory",
        "python_executable",
        "output_directory",
        "log_path",
        "execution_class",
        "submission_state",
    }
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    receipt = create_submission_receipt(
        receipt_path=tmp_path / "receipt.json",
        launch_record_path=record,
        unit_snapshot=snapshot,
        systemd_run_exit_code=1,
    )
    assert receipt["artifact_type"] == "p7c4b2e_target_canary_submission_receipt"
    assert receipt["schema_version"] == 1
    assert set(receipt) == {
        "schema_version",
        "artifact_type",
        "created_at",
        "launch_record_path",
        "launch_record_sha256",
        "systemd_unit",
        "systemd_run_exit_code",
        "unit_snapshot",
        "submission_state",
    }


def test_helper_rejects_malformed_inputs_and_snapshot_schema(tmp_path):
    record, _value = _launch(tmp_path)
    bad = _input(tmp_path / "bad.json", "not json")
    with pytest.raises(Exception, match="evidence_input_invalid"):
        create_launch_record(
            record_path=tmp_path / "other.json",
            git_commit="a" * 40,
            operator_identity="operator",
            authorization_path=bad,
            environment_path=bad,
            proposal_path=bad,
            unit="unit",
            argv=["python"],
            working_directory="/srv",
            python_executable="/srv/python",
            output_directory="/srv/artifacts/out",
            log_path="/secure/log",
        )
    with pytest.raises(Exception, match="unit_snapshot_schema_mismatch"):
        create_submission_receipt(
            receipt_path=tmp_path / "bad-receipt.json",
            launch_record_path=record,
            unit_snapshot={},
            systemd_run_exit_code=0,
        )


def test_atomic_write_failure_leaves_no_final_record(tmp_path, monkeypatch):
    import creditrep.experiments.p7c4b2e_operations_cli as operations

    def fail_link(*_args):
        raise OSError("link failed")

    monkeypatch.setattr(operations.os, "link", fail_link)
    auth = _input(tmp_path / "authorization.json", "{}")
    with pytest.raises(Exception, match="operational_evidence_write_failed"):
        create_launch_record(
            record_path=tmp_path / "launch.json",
            git_commit="a" * 40,
            operator_identity="operator",
            authorization_path=auth,
            environment_path=auth,
            proposal_path=auth,
            unit="unit",
            argv=["python"],
            working_directory="/srv",
            python_executable="/srv/python",
            output_directory="/srv/artifacts/out",
            log_path="/secure/log",
        )
    assert not (tmp_path / "launch.json").exists()
    assert not list(tmp_path.glob(".launch.json.*.tmp"))


def test_resume_precheck_fails_closed_when_required_run_artifacts_are_absent(tmp_path):
    value = resume_precheck(tmp_path)
    assert value["valid"] is False
    assert "resume_precheck_missing_required_artifact" in value["reason_codes"]


def _typed_inner_launch(
    tmp_path,
    *,
    runner_command="run",
    extra_argv=(),
    working_directory="/srv/repo",
    python_executable="/srv/repo/.venv/bin/python",
    output_directory="/srv/repo/artifacts/p7c4b2b-compute-preflight/run-01",
    authorized_output_directory=None,
    run_id="run-01",
):
    commit = "b" * 40
    authorized_output_directory = authorized_output_directory or output_directory

    def typed(value, field):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        value[field] = hashlib.sha256(encoded).hexdigest()
        return value

    profile_value = typed({"git_commit": commit}, "profile_digest")
    environment_value = typed(
        {
            "execution_stage": "target_inner_fit_projection_preflight",
            "source_git_commit": commit,
            "mode": "cpu_parallel_1",
        },
        "environment_digest",
    )
    proposal_value = typed(
        {
            "execution_stage": "target_inner_fit_projection_preflight",
            "mode": "cpu_parallel_1",
        },
        "proposal_digest",
    )
    authorization_value = typed(
        {
            "execution_stage": "target_inner_fit_projection_preflight",
            "source_git_commit": commit,
            "normalized_output_directory": authorized_output_directory,
            "machine_profile_digest": profile_value["profile_digest"],
            "run_id": run_id,
            "mode": "cpu_parallel_1",
        },
        "authorization_digest",
    )
    profile = _input(
        tmp_path / "profile.json",
        json.dumps(profile_value),
    )
    environment = _input(
        tmp_path / "environment.json",
        json.dumps(environment_value),
    )
    proposal = _input(
        tmp_path / "proposal.json",
        json.dumps(proposal_value),
    )
    authorization = _input(
        tmp_path / "authorization.json",
        json.dumps(authorization_value),
    )
    argv = [
        python_executable,
        "-m",
        "creditrep.experiments.p7c4b2b_cli",
        runner_command,
        "--mode",
        "cpu_parallel_1",
        "--profile",
        str(profile),
        "--target-machine-asserted",
        "--target-environment",
        str(environment),
        "--authorization-proposal",
        str(proposal),
        "--effective-authorization",
        str(authorization),
        "--output-dir",
        output_directory,
    ]
    argv.extend(extra_argv)
    launch_path = tmp_path / "inner-launch.json"
    launch = create_launch_record(
        record_path=launch_path,
        git_commit=commit,
        operator_identity="operator",
        authorization_path=authorization,
        environment_path=environment,
        proposal_path=proposal,
        machine_profile_path=profile,
        unit=f"p7c4b2b-inner-p1-{run_id}",
        argv=argv,
        working_directory=working_directory,
        python_executable=python_executable,
        output_directory=output_directory,
        log_path="/secure/run-01.log",
        execution_stage="target-inner-preflight",
    )
    return launch_path, tmp_path / "inner-receipt.json", launch, profile_value


def _directory_alias(alias, target):
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("platform cannot create directory symlinks")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode:
            pytest.skip("platform cannot create a directory alias")


def test_typed_launch_accepts_canonical_repository_alias(tmp_path):
    physical_repo = tmp_path / "physical-repo"
    physical_repo.mkdir()
    (physical_repo / ".venv" / "bin").mkdir(parents=True)
    (physical_repo / "artifacts" / "p7c4b2b-compute-preflight").mkdir(parents=True)
    canonical_repo = tmp_path / "canonical-repo"
    _directory_alias(canonical_repo, physical_repo)
    logical_output = (
        canonical_repo / "artifacts" / "p7c4b2b-compute-preflight" / "run-01"
    )
    physical_output = (
        physical_repo / "artifacts" / "p7c4b2b-compute-preflight" / "run-01"
    )

    _path, _receipt, launch, _profile = _typed_inner_launch(
        tmp_path,
        working_directory=str(canonical_repo),
        python_executable=str(canonical_repo / ".venv" / "bin" / "python"),
        output_directory=str(logical_output),
        authorized_output_directory=str(physical_output),
    )
    assert launch["working_directory"] == str(canonical_repo)
    assert launch["output_directory"] == str(logical_output)


def test_operational_path_resolution_equivalence_is_platform_independent(
    monkeypatch,
):
    logical = Path("logical-repository/output")
    physical = Path("physical-repository/output")

    def equivalent(path):
        return physical if path in {logical, physical} else path

    monkeypatch.setattr(operations, "_resolve_operational_path", equivalent)
    resolved = operations._resolved_operational_paths(
        {"logical": logical, "physical": physical}
    )
    assert resolved == {"logical": physical, "physical": physical}


def test_operational_path_resolution_rejects_injected_parent_retarget(monkeypatch):
    logical = Path("logical-repository/output")
    physical_one = Path("physical-one/output")
    physical_two = Path("physical-two/output")
    target = physical_one

    def retargeting(path):
        if path == logical:
            return target
        return physical_one

    monkeypatch.setattr(operations, "_resolve_operational_path", retargeting)
    paths = {"logical": logical, "physical": physical_one}
    identities = operations._resolved_operational_paths(paths)
    target = physical_two
    with pytest.raises(OperationsError, match="^operational_identity_mismatch$"):
        operations._revalidate_operational_paths(paths, identities)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("different-physical-output", "operational_identity_mismatch"),
        ("python-outside-venv", "operational_working_directory_mismatch"),
        ("output-outside-root", "operational_working_directory_mismatch"),
        ("output-traversal", "operational_working_directory_mismatch"),
        ("run-id-mismatch", "operational_working_directory_mismatch"),
    ],
)
def test_typed_launch_rejects_invalid_operational_path_identity(tmp_path, case, reason):
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    output_root = repo / "artifacts" / "p7c4b2b-compute-preflight"
    output_root.mkdir(parents=True)
    output = output_root / "run-01"
    authorized_output = output
    python = repo / ".venv" / "bin" / "python"
    run_id = "run-01"
    if case == "different-physical-output":
        authorized_output = tmp_path / "different-repo" / "run-01"
    elif case == "python-outside-venv":
        python = tmp_path / "different-python"
    elif case == "output-outside-root":
        output = repo / "outside" / "run-01"
        authorized_output = output
    elif case == "output-traversal":
        output = output_root / ".." / "outside" / "run-01"
        authorized_output = output
    elif case == "run-id-mismatch":
        run_id = "different-run-id"

    with pytest.raises(OperationsError, match=f"^{reason}$"):
        _typed_inner_launch(
            tmp_path,
            working_directory=str(repo),
            python_executable=str(python),
            output_directory=str(output),
            authorized_output_directory=str(authorized_output),
            run_id=run_id,
        )


def test_typed_launch_rejects_output_symlink_escape(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    output_root = repo / "artifacts" / "p7c4b2b-compute-preflight"
    output_root.mkdir(parents=True)
    outside = tmp_path / "outside" / "run-01"
    outside.mkdir(parents=True)
    logical_output = output_root / "run-01"
    _directory_alias(logical_output, outside)

    with pytest.raises(
        OperationsError, match="^operational_working_directory_mismatch$"
    ):
        _typed_inner_launch(
            tmp_path,
            working_directory=str(repo),
            python_executable=str(repo / ".venv" / "bin" / "python"),
            output_directory=str(logical_output),
            authorized_output_directory=str(outside),
        )


def test_typed_launch_rejects_real_parent_symlink_retarget(tmp_path, monkeypatch):
    physical_one = tmp_path / "physical-one"
    physical_two = tmp_path / "physical-two"
    for repo in (physical_one, physical_two):
        (repo / ".venv" / "bin").mkdir(parents=True)
        (repo / "artifacts" / "p7c4b2b-compute-preflight").mkdir(parents=True)
    canonical = tmp_path / "canonical-repo"
    try:
        canonical.symlink_to(physical_one, target_is_directory=True)
    except OSError:
        pytest.skip("platform cannot create directory symlinks")
    real_revalidator = operations._revalidate_operational_paths

    def retarget_after_binding(paths, identities):
        canonical.unlink()
        canonical.symlink_to(physical_two, target_is_directory=True)
        real_revalidator(paths, identities)

    monkeypatch.setattr(
        operations, "_revalidate_operational_paths", retarget_after_binding
    )
    logical_output = canonical / "artifacts" / "p7c4b2b-compute-preflight" / "run-01"
    authorized_output = (
        physical_one / "artifacts" / "p7c4b2b-compute-preflight" / "run-01"
    )
    with pytest.raises(OperationsError, match="^operational_identity_mismatch$"):
        _typed_inner_launch(
            tmp_path,
            working_directory=str(canonical),
            python_executable=str(canonical / ".venv" / "bin" / "python"),
            output_directory=str(logical_output),
            authorized_output_directory=str(authorized_output),
        )


def test_typed_inner_preflight_launch_and_receipt(tmp_path):
    launch_path, receipt_path, launch, profile_value = _typed_inner_launch(tmp_path)
    assert launch["record_digest"]
    assert (
        launch["machine_profile"]["artifact_digest"] == profile_value["profile_digest"]
    )
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    snapshot["InvocationID"] = "invocation-01"
    claim = create_submission_claim(
        launch_record_path=launch_path,
        receipt_path=receipt_path,
    )
    assert claim["submission_state"] == "claimed_not_submitted"
    receipt = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=snapshot,
        systemd_run_exit_code=0,
        observed_unit="p7c4b2b-inner-p1-run-01",
    )
    assert receipt["submission_state"] == "submitted"
    assert receipt["invocation_id"] == "invocation-01"
    assert receipt["receipt_digest"]
    assert receipt["submission_claim_digest"] == claim["claim_digest"]
    with pytest.raises(Exception, match="duplicate_submission"):
        create_submission_receipt(
            receipt_path=tmp_path / "second-inner-receipt.json",
            launch_record_path=launch_path,
            unit_snapshot=snapshot,
            systemd_run_exit_code=0,
            observed_unit="p7c4b2b-inner-p1-run-01",
        )


def test_typed_submission_claim_has_one_concurrent_winner(tmp_path):
    launch_path, receipt_path, _launch, _profile = _typed_inner_launch(tmp_path)

    def claim():
        try:
            create_submission_claim(
                launch_record_path=launch_path,
                receipt_path=receipt_path,
            )
        except Exception as exc:
            return str(exc)
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: claim(), range(2)))
    assert sorted(results) == ["duplicate_submission", "won"]
    with pytest.raises(Exception, match="duplicate_submission"):
        create_submission_claim(
            launch_record_path=launch_path,
            receipt_path=tmp_path / "different-receipt.json",
        )


def test_typed_failed_submission_receipt_is_deterministic(tmp_path):
    launch_path, receipt_path, _launch, _profile = _typed_inner_launch(tmp_path)
    claim = create_submission_claim(
        launch_record_path=launch_path,
        receipt_path=receipt_path,
    )
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    receipt = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=snapshot,
        systemd_run_exit_code=1,
        observed_unit="p7c4b2b-inner-p1-run-01",
    )
    assert receipt["submission_state"] == "submission_failed"
    assert receipt["invocation_id"] == ""
    assert receipt["submission_claim_digest"] == claim["claim_digest"]


def test_typed_resume_launch_is_bound_to_resume_command(tmp_path):
    _path, _receipt, launch, _profile = _typed_inner_launch(
        tmp_path, runner_command="resume"
    )
    assert launch["runner_command"] == "resume"
    assert launch["argv"][3] == "resume"


def test_typed_launch_rejects_unrecorded_scope_flags(tmp_path):
    with pytest.raises(Exception, match="operational_argv_mismatch"):
        _typed_inner_launch(tmp_path, extra_argv=("--fixture",))


def test_unknown_typed_execution_stage_is_rejected(tmp_path):
    record, _value = _launch(tmp_path)
    with pytest.raises(Exception, match="unknown_execution_stage"):
        create_launch_record(
            record_path=record.with_name("unknown.json"),
            git_commit="a" * 40,
            operator_identity="operator",
            authorization_path=tmp_path / "authorization.json",
            environment_path=tmp_path / "environment.json",
            proposal_path=tmp_path / "proposal.json",
            unit="unit",
            argv=["python"],
            working_directory="/srv",
            python_executable="python",
            output_directory="/srv/out",
            log_path="/secure/log",
            execution_stage="unknown",
        )
