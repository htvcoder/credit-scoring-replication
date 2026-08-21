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
    create_submission_result,
    main,
    resume_precheck,
)


RUNBOOK = Path(__file__).parents[1] / "docs" / "P7C4B2B_SINGLE_VM_PREFLIGHT_RUNBOOK.md"
OUTER_RUNBOOK = (
    Path(__file__).parents[1]
    / "docs"
    / "P7C4B2C_OUTER_REFIT_OVERHEAD_PREFLIGHT_RUNBOOK.md"
)


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


def test_b2b_runbook_uses_canonical_submission_and_preserves_exact_argv():
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

    runbook_text = RUNBOOK.read_text(encoding="utf-8")
    assert runbook_text.count("submit-systemd-run ") == 2
    assert "\nsystemd-run --user" not in runbook_text
    assert "systemctl --output=json" not in runbook_text
    assert not {"--fixture", "--max-tasks", "--timeout-seconds"}.intersection(
        token for argv in constructions.values() for token in argv
    )


def test_outer_runbook_jq_argv_matches_exact_submitted_commands():
    commands = list(_logical_shell_commands(OUTER_RUNBOOK.read_text(encoding="utf-8")))
    constructions = {
        command.split("=", 1)[0]: _runbook_argv(command)
        for command in commands
        if "jq -cn --args" in command
    }
    expected_run = [
        "$PYTHON",
        "-m",
        "creditrep.experiments.p7c4b2c_cli",
        "run",
        "--execution-class",
        "target_preflight",
        "--mode",
        "$MODE_P1",
        "--output",
        "$OUT_P1",
        "--target-environment",
        "$ENV_P1",
        "--authorization-proposal",
        "$PROPOSAL_P1",
        "--effective-authorization",
        "$AUTH_P1",
    ]
    expected_resume = [
        *expected_run[:3],
        "resume",
        "--run-dir",
        "$OUT_P1",
        *expected_run[10:],
    ]
    assert constructions == {
        "ARGV_P1": expected_run,
        "RESUME_ARGV_P1": expected_resume,
    }
    runbook_text = OUTER_RUNBOOK.read_text(encoding="utf-8")
    assert runbook_text.count("submit-systemd-run --submission-result") == 2
    assert "\nsystemd-run --user" not in runbook_text
    assert "systemctl --output=json" not in OUTER_RUNBOOK.read_text(encoding="utf-8")


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
    log_path=None,
    unit=None,
):
    commit = "b" * 40
    authorized_output_directory = authorized_output_directory or output_directory
    log_path = log_path or f"/secure/{run_id}.log"

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
        unit=unit or f"p7c4b2b-inner-p1-{run_id}",
        argv=argv,
        working_directory=working_directory,
        python_executable=python_executable,
        output_directory=output_directory,
        log_path=log_path,
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


def test_typed_inner_preflight_receipt_requires_durable_result(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, profile_value = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    assert launch["record_digest"]
    assert (
        launch["machine_profile"]["artifact_digest"] == profile_value["profile_digest"]
    )
    claim = create_submission_claim(
        launch_record_path=launch_path,
        receipt_path=receipt_path,
    )
    assert claim["submission_state"] == "claimed_not_submitted"
    assert claim["submission_state"] == "claimed_not_submitted"
    with pytest.raises(OperationsError, match="^submission_result_missing$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot={field: "" for field in UNIT_SNAPSHOT_FIELDS},
            systemd_run_exit_code=0,
            observed_unit="p7c4b2b-inner-p1-run-01",
        )


def test_typed_submission_claim_has_one_concurrent_winner(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])

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


def test_typed_failed_inner_receipt_requires_durable_result(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    claim = create_submission_claim(
        launch_record_path=launch_path,
        receipt_path=receipt_path,
    )
    assert claim["submission_state"] == "claimed_not_submitted"
    with pytest.raises(OperationsError, match="^submission_result_missing$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot={field: "" for field in UNIT_SNAPSHOT_FIELDS},
            systemd_run_exit_code=1,
            observed_unit="p7c4b2b-inner-p1-run-01",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda launch: launch["argv"].__setitem__(3, "resume"),
        lambda launch: (
            launch.__setitem__("python_executable", "/srv/other/.venv/bin/python"),
            launch["argv"].__setitem__(0, "/srv/other/.venv/bin/python"),
        ),
        lambda launch: (
            launch.__setitem__("output_directory", "/srv/repo/artifacts/p7c4b2b-compute-preflight/other"),
            launch["argv"].__setitem__(-1, "/srv/repo/artifacts/p7c4b2b-compute-preflight/other"),
        ),
        lambda launch: launch.__setitem__("log_path", "/secure/other.log"),
        lambda launch: launch["machine_profile"].__setitem__("artifact_digest", "0" * 64),
        lambda launch: launch.__setitem__("run_id", "other-run"),
        lambda launch: launch.__setitem__("source_git_commit", "c" * 40),
    ],
)
def test_b2b_claim_rejects_redigested_semantic_launch_mutation(
    tmp_path, monkeypatch, mutation
):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: "b" * 40)
    mutation(launch)
    launch["record_digest"] = operations._record_digest(launch, "record_digest")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    with pytest.raises(OperationsError):
        create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)


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


def _typed_outer_launch(
    tmp_path,
    *,
    mode="cpu_parallel_1",
    runner_command="run",
    argv_mutation=None,
    output_directory=None,
    authorized_output_directory=None,
    working_directory=None,
    unit=None,
    advance_git_head=False,
    sort_control_json=False,
):
    repo = Path(working_directory) if working_directory else tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts").mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = (
        Path(output_directory) if output_directory else repo / "artifacts" / "outer-run"
    )
    authorized_output = (
        Path(authorized_output_directory) if authorized_output_directory else output
    )
    if runner_command == "resume":
        output.mkdir(parents=True, exist_ok=True)
    python = repo / ".venv" / "bin" / "python"
    task_ids = [f"task-{index:03d}" for index in range(162)]
    dataset_ids = ["AC", "GC", "TH02", "HMEQ", "TC", "GMC"]
    dataset_hashes = {
        dataset_id: f"{index + 1:064x}" for index, dataset_id in enumerate(dataset_ids)
    }
    dataset_binding = "d" * 64

    def typed(value, field):
        value[field] = operations._record_digest(value, field)
        return value

    environment_value = typed(
        {
            "git_commit": commit,
            "execution_mode": mode,
            "output_directory": str(authorized_output),
            "dataset_ids": dataset_ids,
            "dataset_hashes": dataset_hashes,
            "dataset_binding_digest": dataset_binding,
        },
        "environment_digest",
    )
    proposal_value = typed(
        {
            "execution_stage": "target_projection_preflight",
            "git_commit": commit,
            "execution_mode": mode,
            "output_directory": str(authorized_output),
            "target_environment_digest": environment_value["environment_digest"],
            "task_ids": task_ids,
            "maximum_task_count": 162,
            "dataset_ids": dataset_ids,
            "dataset_hashes": dataset_hashes,
            "dataset_binding_digest": dataset_binding,
        },
        "proposal_digest",
    )
    authorization_value = typed(
        {
            "execution_stage": "target_projection_preflight",
            "git_commit": commit,
            "execution_mode": mode,
            "output_directory": str(authorized_output),
            "target_environment_digest": environment_value["environment_digest"],
            "proposal_digest": proposal_value["proposal_digest"],
            "task_ids": task_ids,
            "maximum_task_count": 162,
            "dataset_ids": dataset_ids,
            "dataset_hashes": dataset_hashes,
            "dataset_binding_digest": dataset_binding,
        },
        "authorization_digest",
    )
    environment = _input(
        tmp_path / f"environment-{mode}.json",
        json.dumps(environment_value, sort_keys=sort_control_json),
    )
    proposal = _input(
        tmp_path / f"proposal-{mode}.json",
        json.dumps(proposal_value, sort_keys=sort_control_json),
    )
    authorization = _input(
        tmp_path / f"authorization-{mode}.json",
        json.dumps(authorization_value, sort_keys=sort_control_json),
    )
    controls = [
        "--target-environment",
        str(environment),
        "--authorization-proposal",
        str(proposal),
        "--effective-authorization",
        str(authorization),
    ]
    prefix = [str(python), "-m", "creditrep.experiments.p7c4b2c_cli", runner_command]
    argv = (
        [
            *prefix,
            "--execution-class",
            "target_preflight",
            "--mode",
            mode,
            "--output",
            str(output),
            *controls,
        ]
        if runner_command == "run"
        else [*prefix, "--run-dir", str(output), *controls]
    )
    if argv_mutation:
        argv_mutation(argv)
    mode_token = "p1" if mode == "cpu_parallel_1" else "p2"
    original_launch_path = None
    if runner_command == "resume":
        original_launch_path = tmp_path / f"outer-{mode_token}-original-launch.json"
        original_launch = {
            "schema_version": 1,
            "artifact_type": "p7c4b2c_target_outer_projection_preflight_launch_record",
            "execution_stage": "target-outer-projection-preflight",
            "runner_command": "run",
            "mode": mode,
            "source_git_commit": commit,
            "resolved_output_directory": str(output.resolve()),
            "systemd_unit": f"p7c4b2c-outer-{mode_token}-outer-run-initial.service",
            "environment": {"artifact_digest": environment_value["environment_digest"]},
            "proposal": {"artifact_digest": proposal_value["proposal_digest"]},
            "authorization": {
                "artifact_digest": authorization_value["authorization_digest"]
            },
        }
        original_launch["record_digest"] = operations._record_digest(
            original_launch, "record_digest"
        )
        _input(original_launch_path, json.dumps(original_launch))
    launch_path = tmp_path / f"outer-{mode_token}-{runner_command}-launch.json"
    if advance_git_head:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "advance",
            ],
            check=True,
        )
    selected_unit = unit or (
        f"p7c4b2c-outer-{mode_token}-{output.name}-{runner_command}.service"
    )
    launch = create_launch_record(
        record_path=launch_path,
        git_commit=commit,
        operator_identity="operator",
        authorization_path=authorization,
        environment_path=environment,
        proposal_path=proposal,
        unit=selected_unit,
        argv=argv,
        working_directory=str(repo),
        python_executable=str(python),
        output_directory=str(output),
        log_path=str(
            repo / "artifacts" / "p7c4b2c-operations" / f"{selected_unit}.log"
        ),
        execution_stage="target-outer-projection-preflight",
        resume_of_launch_record_path=original_launch_path,
    )
    return (
        launch_path,
        tmp_path / f"outer-{mode_token}-{runner_command}-receipt.json",
        launch,
        argv,
    )


def _outer_submission_result(
    tmp_path, launch_path, launch, *, exit_code=0, stdout=None, stderr=""
):
    operations._create_submission_attempt(launch_path)
    invocation_id = "09c753f133ac4a3fae89ba13ec21b3fe"
    reported_unit = operations._expected_systemd_reported_unit(
        launch.get("execution_stage"), launch["systemd_unit"]
    )
    stdout_path = tmp_path / "systemd-run.stdout"
    stderr_path = tmp_path / "systemd-run.stderr"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(
        stdout
        if stdout is not None
        else (
            f"Running as unit: {reported_unit}; invocation ID: "
            f"{invocation_id}\n"
            if exit_code == 0
            else "Failed to start transient unit\n"
        ),
        encoding="utf-8",
    )
    stderr_path.write_text(stderr, encoding="utf-8")
    exit_code_path = tmp_path / "systemd-run.exit-code"
    exit_code_path.write_bytes(f"{exit_code}\n".encode("ascii"))
    operations._create_submission_capture(
        launch_record_path=launch_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        exit_code_path=exit_code_path,
        systemd_run_exit_code=exit_code,
    )
    result_path = tmp_path / "submission-result.json"
    result = create_submission_result(
        result_path=result_path,
        launch_record_path=launch_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        exit_code_path=exit_code_path,
        systemd_run_exit_code=exit_code,
    )
    return result_path, result


@pytest.mark.parametrize("mode", ["cpu_parallel_1", "cpu_parallel_2"])
def test_typed_outer_projection_run_launch_is_exact_and_bound(tmp_path, mode):
    _path, _receipt, launch, argv = _typed_outer_launch(tmp_path, mode=mode)
    assert launch["protocol_stage"] == "target_projection_preflight"
    assert launch["mode"] == mode
    assert launch["argv"] == argv
    assert launch["task_set_digest"] == operations._canonical_digest(launch["task_ids"])
    assert launch["resolved_output_directory"] == str(
        Path(launch["output_directory"]).resolve()
    )


@pytest.mark.parametrize("runner_command", ["run", "resume"])
def test_typed_outer_projection_accepts_sorted_control_json(tmp_path, runner_command):
    _path, _receipt, launch, _argv = _typed_outer_launch(
        tmp_path,
        runner_command=runner_command,
        sort_control_json=True,
    )
    assert launch["protocol_stage"] == "target_projection_preflight"
    assert launch["mode"] == "cpu_parallel_1"
    assert len(launch["task_ids"]) == 162


def _outer_identity_controls():
    dataset_ids = ["AC", "GC", "TH02", "HMEQ", "TC", "GMC"]
    dataset_hashes = {
        dataset_id: f"{index + 1:064x}" for index, dataset_id in enumerate(dataset_ids)
    }
    task_ids = [f"task-{index:03d}" for index in range(162)]
    environment = {
        "git_commit": "a" * 40,
        "execution_mode": "cpu_parallel_1",
        "environment_digest": "e" * 64,
        "dataset_ids": dataset_ids,
        "dataset_hashes": dataset_hashes,
        "dataset_binding_digest": "d" * 64,
    }
    proposal = {
        "execution_stage": "target_projection_preflight",
        "git_commit": "a" * 40,
        "execution_mode": "cpu_parallel_1",
        "target_environment_digest": environment["environment_digest"],
        "proposal_digest": "p" * 64,
        "task_ids": task_ids,
        "maximum_task_count": 162,
        "dataset_ids": dataset_ids,
        "dataset_hashes": dataset_hashes,
        "dataset_binding_digest": "d" * 64,
    }
    authorization = {
        **proposal,
        "authorization_digest": "b" * 64,
    }
    return environment, proposal, authorization


def test_outer_control_identity_accepts_json_round_trip_and_reordered_hash_mapping():
    environment, proposal, authorization = _outer_identity_controls()
    environment, proposal, authorization = (
        json.loads(json.dumps(value, sort_keys=True))
        for value in (environment, proposal, authorization)
    )
    mode, _task_set_digest, task_ids = operations._outer_control_identity(
        authorization=authorization,
        environment=environment,
        proposal=proposal,
        git_commit="a" * 40,
    )
    assert (mode, len(task_ids)) == ("cpu_parallel_1", 162)

    reordered = dict(reversed(list(authorization["dataset_hashes"].items())))
    for control in (environment, proposal, authorization):
        control["dataset_hashes"] = reordered
    assert operations._outer_control_identity(
        authorization=authorization,
        environment=environment,
        proposal=proposal,
        git_commit="a" * 40,
    )[0] == "cpu_parallel_1"


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "malformed", "reordered_ids", "hash_disagreement", "binding_disagreement"],
)
def test_outer_control_identity_keeps_dataset_contract_strict(mutation):
    environment, proposal, authorization = _outer_identity_controls()
    if mutation == "missing":
        for control in (environment, proposal, authorization):
            control["dataset_hashes"] = {
                key: value
                for key, value in control["dataset_hashes"].items()
                if key != "AC"
            }
    elif mutation == "extra":
        for control in (environment, proposal, authorization):
            control["dataset_hashes"] = {
                **control["dataset_hashes"],
                "EXTRA": "0" * 64,
            }
    elif mutation == "malformed":
        for control in (environment, proposal, authorization):
            control["dataset_hashes"] = {
                **control["dataset_hashes"],
                "AC": "not-a-sha256",
            }
    elif mutation == "reordered_ids":
        for control in (environment, proposal, authorization):
            control["dataset_ids"] = list(reversed(control["dataset_ids"]))
    elif mutation == "hash_disagreement":
        proposal["dataset_hashes"] = {
            **proposal["dataset_hashes"],
            "AC": "0" * 64,
        }
    else:
        proposal["dataset_binding_digest"] = "c" * 64
    with pytest.raises(OperationsError, match="^operational_identity_mismatch$"):
        operations._outer_control_identity(
            authorization=authorization,
            environment=environment,
            proposal=proposal,
            git_commit="a" * 40,
        )


def test_typed_outer_projection_rejects_stale_control_source_sha(tmp_path):
    with pytest.raises(OperationsError, match="^operational_git_provenance_mismatch$"):
        _typed_outer_launch(tmp_path, advance_git_head=True)


def test_historical_outer_schema_v1_launch_claim_receipt_remains_compatible(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    launch["schema_version"] = 1
    launch["record_digest"] = operations._record_digest(launch, "record_digest")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    claim = create_submission_claim(
        launch_record_path=launch_path, receipt_path=receipt_path
    )
    assert claim["schema_version"] == 1
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    snapshot["InvocationID"] = "historical-valid-invocation"
    receipt = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=snapshot,
        systemd_run_exit_code=0,
        observed_unit=launch["systemd_unit"],
    )
    assert receipt["schema_version"] == 1
    assert receipt["invocation_id"] == "historical-valid-invocation"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda argv: argv.__setitem__(2, "creditrep.experiments.p7c4b2b_cli"),
        lambda argv: argv.__setitem__(3, "resume"),
        lambda argv: argv.__setitem__(5, "synthetic_validation"),
        lambda argv: argv.extend(["--max-samples", "1"]),
        lambda argv: argv.extend(["--unknown", "value"]),
        lambda argv: argv.__setitem__(6, "--output"),
    ],
)
def test_typed_outer_projection_rejects_nonexact_argv(tmp_path, mutation):
    with pytest.raises(OperationsError, match="^operational_argv_mismatch$"):
        _typed_outer_launch(tmp_path, argv_mutation=mutation)


def test_typed_outer_projection_resume_has_fresh_operational_identity(tmp_path):
    launch_path, receipt_path, launch, argv = _typed_outer_launch(
        tmp_path, runner_command="resume"
    )
    assert launch["runner_command"] == "resume"
    assert launch["resume_of_launch_record"]["systemd_unit"].endswith(
        "-initial.service"
    )
    assert argv[4:6] == ["--run-dir", launch["output_directory"]]
    claim = create_submission_claim(
        launch_record_path=launch_path, receipt_path=receipt_path
    )
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)
    snapshot["InvocationID"] = result["invocation_id"]
    receipt = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=snapshot,
        systemd_run_exit_code=0,
        observed_unit=launch["systemd_unit"],
        submission_result_path=result_path,
    )
    assert receipt["submission_claim_digest"] == claim["claim_digest"]
    assert receipt["launch_record_digest"] == launch["record_digest"]


def test_typed_outer_projection_resume_rejects_reused_unit(tmp_path):
    with pytest.raises(OperationsError, match="^resume_launch_identity_mismatch$"):
        _typed_outer_launch(
            tmp_path,
            runner_command="resume",
            unit="p7c4b2c-outer-p1-outer-run-initial.service",
        )


def test_typed_outer_projection_claim_is_atomic_and_receipt_path_bound(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                create_submission_claim,
                launch_record_path=launch_path,
                receipt_path=receipt_path,
            )
            for _ in range(8)
        ]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result()["submission_state"])
        except OperationsError as exc:
            outcomes.append(str(exc))
    assert outcomes.count("claimed_not_submitted") == 1
    assert outcomes.count("duplicate_submission") == 7
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    result_path, _result = _outer_submission_result(
        tmp_path, launch_path, launch, exit_code=1
    )
    with pytest.raises(OperationsError, match="^duplicate_submission$"):
        create_submission_receipt(
            receipt_path=tmp_path / "alternate-receipt.json",
            launch_record_path=launch_path,
            unit_snapshot=snapshot,
            systemd_run_exit_code=1,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


def test_typed_outer_projection_failed_submission_is_evidence_and_success_requires_invocation(
    tmp_path,
):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    result_path, _result = _outer_submission_result(
        tmp_path, launch_path, launch, exit_code=1
    )
    failed = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=snapshot,
        systemd_run_exit_code=1,
        observed_unit=launch["systemd_unit"],
        submission_result_path=result_path,
    )
    assert failed["submission_state"] == "submission_failed"
    assert failed["invocation_id"] is None

    second_path, second_receipt, second, _argv = _typed_outer_launch(
        tmp_path / "second"
    )
    create_submission_claim(launch_record_path=second_path, receipt_path=second_receipt)
    with pytest.raises(
        OperationsError, match="^systemd_run_invocation_id_count_invalid$"
    ):
        _outer_submission_result(
            tmp_path / "second",
            second_path,
            second,
            stdout=f"Running as unit: {second['systemd_unit']};\n",
        )


def test_outer_receipt_live_and_collected_recovery_are_submission_only(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    snapshot.update(
        {
            "LoadState": "loaded",
            "ActiveState": "active",
            "InvocationID": result["invocation_id"],
        }
    )
    receipt = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=snapshot,
        systemd_run_exit_code=0,
        observed_unit=launch["systemd_unit"],
        submission_result_path=result_path,
    )
    assert receipt["unit_snapshot_status"] == "live_verified"
    assert receipt["evidence_scope"] == "submission_outcome_only_not_compute_success"
    assert "compute_success" not in receipt

    second_path, second_receipt, second, _argv = _typed_outer_launch(
        tmp_path / "collected"
    )
    create_submission_claim(launch_record_path=second_path, receipt_path=second_receipt)
    second_result_path, _result = _outer_submission_result(
        tmp_path / "collected", second_path, second
    )
    collected = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    collected["LoadState"] = "not-found"
    recovered = create_submission_receipt(
        receipt_path=second_receipt,
        launch_record_path=second_path,
        unit_snapshot=collected,
        systemd_run_exit_code=0,
        observed_unit=second["systemd_unit"],
        submission_result_path=second_result_path,
    )
    assert recovered["unit_snapshot_status"] == "unavailable_unit_collected"
    assert recovered["invocation_id"] == "09c753f133ac4a3fae89ba13ec21b3fe"


def test_outer_submit_rejects_output_created_before_submission_attempt(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    Path(launch["output_directory"]).mkdir()

    with pytest.raises(OperationsError, match="^output_collision$"):
        operations._create_submission_attempt(launch_path)


def test_outer_receipt_accepts_runner_created_authorized_output_after_submission(
    tmp_path, monkeypatch
):
    """Reproduce launch -> claim -> submit -> output -> receipt ordering."""
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)

    output = Path(launch["output_directory"])
    output.mkdir()
    (output / "partial-run-artifact.json").write_text("partial\n", encoding="utf-8")
    # A receipt validates immutable execution evidence, not the mutable HEAD
    # after a hotfix checkout.
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: "b" * 40)
    collected = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    collected["LoadState"] = "not-found"

    receipt = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=collected,
        systemd_run_exit_code=0,
        observed_unit=launch["systemd_unit"],
        submission_result_path=result_path,
    )

    assert receipt["invocation_id"] == result["invocation_id"]
    assert receipt["unit_snapshot_status"] == "unavailable_unit_collected"
    assert receipt["evidence_scope"] == "submission_outcome_only_not_compute_success"


def test_outer_receipt_still_rejects_substituted_output_identity(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, _result = _outer_submission_result(tmp_path, launch_path, launch)
    launch["output_directory"] = str(tmp_path / "other-output")
    launch["record_digest"] = operations._record_digest(launch, "record_digest")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    collected = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    collected["LoadState"] = "not-found"

    with pytest.raises(OperationsError, match="^operational_identity_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=collected,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


def test_b2b_receipt_recovery_does_not_depend_on_current_git_head(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(
        operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"]
    )
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, _result = _outer_submission_result(tmp_path, launch_path, launch)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: "b" * 40)

    receipt = create_submission_receipt(
        receipt_path=receipt_path,
        launch_record_path=launch_path,
        unit_snapshot=None,
        systemd_run_exit_code=0,
        observed_unit=launch["systemd_unit"],
        submission_result_path=result_path,
    )

    assert receipt["unit_snapshot_status"] == "recovered_from_immutable_submission_result"


@pytest.mark.parametrize(
    "field,value",
    [("mode", "cpu_parallel_2"), ("source_git_commit", "c" * 40)],
)
def test_outer_receipt_still_rejects_modified_launch_control_binding(
    tmp_path, field, value
):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, _result = _outer_submission_result(tmp_path, launch_path, launch)
    launch[field] = value
    launch["record_digest"] = operations._record_digest(launch, "record_digest")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    collected = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    collected["LoadState"] = "not-found"

    with pytest.raises(OperationsError, match="^operational_identity_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=collected,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


def test_exact_incident_systemd_output_with_service_suffix_parses():
    unit = "p7c4b2c-outer-p1-target-outer-p1-20260815-01.service"
    output = (
        f"Running as unit: {unit}; invocation ID: 09c753f133ac4a3fae89ba13ec21b3fe\n"
    )
    assert (
        operations._parse_systemd_run_output(
            output, "", unit, "target-outer-projection-preflight"
        )
        == "09c753f133ac4a3fae89ba13ec21b3fe"
    )


@pytest.mark.parametrize(
    "stdout,stderr,expected",
    [
        (
            "",
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n",
            "09c753f133ac4a3fae89ba13ec21b3fe",
        ),
        (
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n",
            "",
            "09c753f133ac4a3fae89ba13ec21b3fe",
        ),
    ],
)
def test_b2b_systemd_parser_accepts_one_identity_from_either_channel(
    stdout, stderr, expected
):
    assert operations._parse_systemd_run_output(
        stdout, stderr, "p7c4b2b-inner-p1-run-01", "target-inner-preflight"
    ) == expected


def test_systemd_parser_allows_only_identical_cross_channel_duplicate():
    identity = (
        "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
        "09c753f133ac4a3fae89ba13ec21b3fe\n"
    )
    assert operations._parse_systemd_run_output(
        identity, identity, "p7c4b2b-inner-p1-run-01", "target-inner-preflight"
    ) == "09c753f133ac4a3fae89ba13ec21b3fe"


@pytest.mark.parametrize(
    "stdout,stderr,reason",
    [
        (
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n",
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "19c753f133ac4a3fae89ba13ec21b3fe\n",
            "systemd_run_output_conflict",
        ),
        (
            "",
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n"
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "19c753f133ac4a3fae89ba13ec21b3fe\n",
            "systemd_run_invocation_id_count_invalid",
        ),
        (
            "",
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: nope\n",
            "invocation_id_malformed",
        ),
        (
            "",
            "Running as unit: wrong.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n",
            "systemd_unit_mismatch",
        ),
    ],
)
def test_b2b_systemd_parser_fails_closed_for_ambiguous_or_invalid_evidence(
    stdout, stderr, reason
):
    with pytest.raises(OperationsError, match=f"^{reason}$"):
        operations._parse_systemd_run_output(
            stdout, stderr, "p7c4b2b-inner-p1-run-01", "target-inner-preflight"
        )


@pytest.mark.parametrize(
    "stdout,stderr",
    [
        (
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n",
            "Running as unit: foo.service\n",
        ),
        (
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n",
            "invocation ID: 19c753f133ac4a3fae89ba13ec21b3fe\n",
        ),
        ("", "Running as unit:\n"),
        (
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n",
            "Running as unit: foo.service; invocation ID: nope\n",
        ),
        (
            "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\nRunning as unit: foo.service\n",
            "",
        ),
    ],
)
def test_b2b_systemd_parser_rejects_unaccounted_identity_markers(stdout, stderr):
    with pytest.raises(
        OperationsError,
        match="^(systemd_run_invocation_id_count_invalid|invocation_id_malformed)$",
    ):
        operations._parse_systemd_run_output(
            stdout, stderr, "p7c4b2b-inner-p1-run-01", "target-inner-preflight"
        )


def test_b2b_systemd_parser_allows_unrelated_stderr_noise():
    assert operations._parse_systemd_run_output(
        "Running as unit: p7c4b2b-inner-p1-run-01.service; invocation ID: "
        "09c753f133ac4a3fae89ba13ec21b3fe\n",
        "notice: user manager diagnostic\n",
        "p7c4b2b-inner-p1-run-01",
        "target-inner-preflight",
    ) == "09c753f133ac4a3fae89ba13ec21b3fe"


def test_typed_inner_launch_rejects_service_suffixed_logical_unit(tmp_path):
    with pytest.raises(OperationsError, match="^systemd_unit_mismatch$"):
        _typed_inner_launch(tmp_path, unit="p7c4b2b-inner-p1-run-01.service")


def test_b2b_cli_reconstructs_real_systemd_evidence_without_resubmission(
    tmp_path, monkeypatch
):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(
        operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"]
    )
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    operations._create_submission_attempt(launch_path)
    stdout_path = tmp_path / "systemd-run.stdout"
    stderr_path = tmp_path / "systemd-run.stderr"
    exit_path = tmp_path / "systemd-run.exit"
    result_path = tmp_path / "submission-result.json"
    stdout_path.write_bytes(b"")
    stderr_path.write_bytes(
        f"Running as unit: {launch['systemd_unit']}.service; invocation ID: "
        "09c753f133ac4a3fae89ba13ec21b3fe\n".encode("ascii")
    )
    exit_path.write_bytes(b"0\n")
    operations._create_submission_capture(
        launch_record_path=launch_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        exit_code_path=exit_path,
        systemd_run_exit_code=0,
    )
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("systemd-run must not be called during reconstruction")
        ),
    )
    assert main([
        "create-submission-result", "--submission-result", str(result_path),
        "--launch-record", str(launch_path), "--systemd-run-stdout", str(stdout_path),
        "--systemd-run-stderr", str(stderr_path), "--systemd-run-exit-code-file", str(exit_path),
        "--systemd-run-exit-code", "0",
    ]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["artifact_type"] == "p7c4b2b_target_inner_preflight_submission_result"
    assert result["invocation_id"] == "09c753f133ac4a3fae89ba13ec21b3fe"
    assert result["stdout_sha256"] == hashlib.sha256(stdout_path.read_bytes()).hexdigest()
    assert result["stderr_sha256"] == hashlib.sha256(stderr_path.read_bytes()).hexdigest()
    assert result["exit_code_sha256"] == hashlib.sha256(exit_path.read_bytes()).hexdigest()
    assert main([
        "create-submission-receipt", "--receipt", str(receipt_path),
        "--launch-record", str(launch_path), "--unit", launch["systemd_unit"],
        "--submission-result", str(result_path), "--systemd-run-exit-code", "0",
    ]) == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["invocation_id"] == result["invocation_id"]
    assert receipt["unit_snapshot_status"] == "recovered_from_immutable_submission_result"
    with pytest.raises(OperationsError, match="^submission_already_attempted$"):
        operations.submit_systemd_run(
            result_path=tmp_path / "retry-result.json", launch_record_path=launch_path,
            stdout_path=tmp_path / "retry.stdout", stderr_path=tmp_path / "retry.stderr",
            exit_code_path=tmp_path / "retry.exit",
        )


@pytest.mark.parametrize("files", [("stdout",), ("stdout", "stderr")])
def test_b2b_partial_raw_evidence_fails_closed_and_blocks_resubmit(
    tmp_path, monkeypatch, files
):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(
        operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"]
    )
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    operations._create_submission_attempt(launch_path)
    paths = {
        "stdout": tmp_path / "stdout", "stderr": tmp_path / "stderr", "exit": tmp_path / "exit",
    }
    for name in files:
        paths[name].write_bytes(b"")
    with pytest.raises(OperationsError):
        create_submission_result(
            result_path=tmp_path / "result.json", launch_record_path=launch_path,
            stdout_path=paths["stdout"], stderr_path=paths["stderr"],
            exit_code_path=paths["exit"], systemd_run_exit_code=0,
        )
    with pytest.raises(OperationsError, match="^submission_already_attempted$"):
        operations.submit_systemd_run(
            result_path=tmp_path / "retry-result.json", launch_record_path=launch_path,
            stdout_path=tmp_path / "retry.stdout", stderr_path=tmp_path / "retry.stderr",
            exit_code_path=tmp_path / "retry.exit",
        )


@pytest.mark.parametrize("tampered", ["stdout", "stderr", "exit"])
def test_b2b_capture_rejects_pre_result_raw_evidence_tampering(
    tmp_path, monkeypatch, tampered
):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(
        operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"]
    )
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    operations._create_submission_attempt(launch_path)
    paths = {
        "stdout": tmp_path / "stdout", "stderr": tmp_path / "stderr", "exit": tmp_path / "exit",
    }
    paths["stdout"].write_bytes(b"")
    paths["stderr"].write_bytes((
        f"Running as unit: {launch['systemd_unit']}.service; invocation ID: "
        "701a0197010141c8b3a1aa97857576b7\n"
    ).encode("ascii"))
    paths["exit"].write_bytes(b"0\n")
    capture = operations._create_submission_capture(
        launch_record_path=launch_path,
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        exit_code_path=paths["exit"],
        systemd_run_exit_code=0,
    )
    assert capture["artifact_type"] == "p7c4b2b_target_inner_preflight_submission_capture"
    if tampered == "stdout":
        paths[tampered].write_bytes(b"altered\n")
    elif tampered == "stderr":
        paths[tampered].write_bytes((
            f"Running as unit: {launch['systemd_unit']}.service; invocation ID: "
            "09c753f133ac4a3fae89ba13ec21b3fe\n"
        ).encode("ascii"))
    else:
        paths[tampered].write_bytes(b"1\n")
    with pytest.raises(
        OperationsError,
        match="^(submission_capture_invalid|systemd_run_exit_code_evidence_mismatch)$",
    ):
        create_submission_result(
            result_path=tmp_path / "result.json", launch_record_path=launch_path,
            stdout_path=paths["stdout"], stderr_path=paths["stderr"],
            exit_code_path=paths["exit"], systemd_run_exit_code=0,
        )


def test_b2b_complete_raw_evidence_without_capture_cannot_reconstruct(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(
        operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"]
    )
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    operations._create_submission_attempt(launch_path)
    stdout_path, stderr_path, exit_path = tmp_path / "stdout", tmp_path / "stderr", tmp_path / "exit"
    stdout_path.write_bytes(b"")
    stderr_path.write_bytes((
        f"Running as unit: {launch['systemd_unit']}.service; invocation ID: "
        "701a0197010141c8b3a1aa97857576b7\n"
    ).encode("ascii"))
    exit_path.write_bytes(b"0\n")
    with pytest.raises(OperationsError, match="^evidence_input_missing$"):
        create_submission_result(
            result_path=tmp_path / "result.json", launch_record_path=launch_path,
            stdout_path=stdout_path, stderr_path=stderr_path,
            exit_code_path=exit_path, systemd_run_exit_code=0,
        )
    with pytest.raises(OperationsError, match="^submission_already_attempted$"):
        operations.submit_systemd_run(
            result_path=tmp_path / "retry-result.json", launch_record_path=launch_path,
            stdout_path=tmp_path / "retry-stdout", stderr_path=tmp_path / "retry-stderr",
            exit_code_path=tmp_path / "retry-exit",
        )


def test_outer_submit_wrapper_atomically_captures_exact_systemd_result(
    tmp_path, monkeypatch
):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    original_run = subprocess.run
    observed = {}

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        observed["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                f"Running as unit: {launch['systemd_unit']}; invocation ID: "
                "09c753f133ac4a3fae89ba13ec21b3fe\n"
            ).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(operations.subprocess, "run", fake_run)
    result = operations.submit_systemd_run(
        result_path=tmp_path / "submission-result.json",
        launch_record_path=launch_path,
        stdout_path=tmp_path / "systemd-run.stdout",
        stderr_path=tmp_path / "systemd-run.stderr",
        exit_code_path=tmp_path / "systemd-run.exit-code",
    )
    assert observed["command"] == [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={launch['systemd_unit']}",
        f"--working-directory={launch['working_directory']}",
        f"--property=StandardOutput=append:{launch['log_path']}",
        f"--property=StandardError=append:{launch['log_path']}",
        *launch["argv"],
    ]
    assert result["submission_state"] == "submitted"
    assert Path(result["exit_code_path"]).read_text(encoding="ascii") == "0\n"


def test_b2b_submit_wrapper_creates_durable_result_and_recovers_collected_unit(
    tmp_path, monkeypatch
):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0,
            stdout=(f"Running as unit: {launch['systemd_unit']}.service; invocation ID: "
                    "09c753f133ac4a3fae89ba13ec21b3fe\n").encode(),
            stderr=b"",
        )

    monkeypatch.setattr(operations.subprocess, "run", fake_run)
    result_path = tmp_path / "b2b-submission-result.json"
    result = operations.submit_systemd_run(
        result_path=result_path, launch_record_path=launch_path,
        stdout_path=tmp_path / "b2b.stdout", stderr_path=tmp_path / "b2b.stderr",
        exit_code_path=tmp_path / "b2b.exit",
    )
    receipt = create_submission_receipt(
        receipt_path=receipt_path, launch_record_path=launch_path,
        unit_snapshot=None, systemd_run_exit_code=0,
        observed_unit=launch["systemd_unit"], submission_result_path=result_path,
    )
    assert result["artifact_type"] == "p7c4b2b_target_inner_preflight_submission_result"
    capture_path = operations._submission_capture_path(launch_path)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert capture["artifact_type"] == "p7c4b2b_target_inner_preflight_submission_capture"
    assert result["submission_capture_path"] == str(capture_path.resolve())
    assert result["submission_capture_sha256"] == hashlib.sha256(capture_path.read_bytes()).hexdigest()
    assert capture["stdout_sha256"] == result["stdout_sha256"]
    assert capture["stderr_sha256"] == result["stderr_sha256"]
    assert capture["exit_code_sha256"] == result["exit_code_sha256"]
    assert receipt["invocation_id"] == result["invocation_id"]
    assert receipt["unit_snapshot_status"] == "recovered_from_immutable_submission_result"


def test_submission_capture_is_no_clobber_and_receipt_rejects_swapped_capture(
    tmp_path, monkeypatch
):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first_path, first_receipt, first, _profile = _typed_inner_launch(tmp_path / "first")
    second_path, second_receipt, second, _profile = _typed_inner_launch(tmp_path / "second")
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: "b" * 40)
    create_submission_claim(launch_record_path=first_path, receipt_path=first_receipt)
    create_submission_claim(launch_record_path=second_path, receipt_path=second_receipt)
    first_result_path, first_result = _outer_submission_result(tmp_path / "first", first_path, first)
    second_result_path, second_result = _outer_submission_result(tmp_path / "second", second_path, second)
    with pytest.raises(OperationsError, match="^operational_evidence_collision$"):
        operations._create_submission_capture(
            launch_record_path=first_path,
            stdout_path=Path(first_result["stdout_path"]),
            stderr_path=Path(first_result["stderr_path"]),
            exit_code_path=Path(first_result["exit_code_path"]),
            systemd_run_exit_code=0,
        )
    first_capture_path = operations._submission_capture_path(first_path)
    first_capture = json.loads(first_capture_path.read_text(encoding="utf-8"))
    second_result["submission_capture_path"] = str(first_capture_path.resolve())
    second_result["submission_capture_sha256"] = hashlib.sha256(first_capture_path.read_bytes()).hexdigest()
    second_result["submission_capture_digest"] = first_capture["capture_digest"]
    second_result["result_digest"] = operations._record_digest(second_result, "result_digest")
    second_result_path.write_text(json.dumps(second_result), encoding="utf-8")
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=second_receipt, launch_record_path=second_path,
            unit_snapshot=None, systemd_run_exit_code=0,
            observed_unit=second["systemd_unit"], submission_result_path=second_result_path,
        )


def test_b2b_cli_e2e_creates_and_recovers_durable_submission_evidence(
    tmp_path, monkeypatch, capsys
):
    launch_path, receipt_path, template, _profile = _typed_inner_launch(tmp_path)
    launch_path.unlink()
    monkeypatch.setattr(
        operations, "_working_tree_git_head", lambda _path: template["source_git_commit"]
    )

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(f"Running as unit: {template['systemd_unit']}.service; invocation ID: "
                    "09c753f133ac4a3fae89ba13ec21b3fe\n").encode(),
            stderr=b"synthetic stderr\n",
        )

    monkeypatch.setattr(operations.subprocess, "run", fake_run)
    assert main([
        "create-launch-record", "--record", str(launch_path),
        "--git-commit", template["source_git_commit"],
        "--operator-identity", template["operator_identity"],
        "--execution-stage", "target-inner-preflight",
        "--machine-profile", template["machine_profile"]["path"],
        "--authorization", template["authorization"]["path"],
        "--environment", template["environment"]["path"],
        "--proposal", template["proposal"]["path"],
        "--unit", template["systemd_unit"],
        "--argv-json", json.dumps(template["argv"]),
        "--working-directory", template["working_directory"],
        "--python-executable", template["python_executable"],
        "--output-directory", template["output_directory"],
        "--log-path", template["log_path"],
    ]) == 0
    assert main([
        "create-submission-claim", "--launch-record", str(launch_path),
        "--receipt", str(receipt_path),
    ]) == 0
    result_path = tmp_path / "submission-result.json"
    stdout_path = tmp_path / "systemd-run.stdout"
    stderr_path = tmp_path / "systemd-run.stderr"
    exit_path = tmp_path / "systemd-run.exit"
    assert main([
        "submit-systemd-run", "--submission-result", str(result_path),
        "--launch-record", str(launch_path),
        "--systemd-run-stdout", str(stdout_path),
        "--systemd-run-stderr", str(stderr_path),
        "--systemd-run-exit-code-file", str(exit_path),
    ]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert main([
        "create-submission-receipt", "--receipt", str(receipt_path),
        "--launch-record", str(launch_path), "--unit", template["systemd_unit"],
        "--submission-result", str(result_path),
        "--systemd-run-exit-code", str(result["systemd_run_exit_code"]),
    ]) == 0
    capsys.readouterr()
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    claim = json.loads(operations._submission_claim_path(launch_path).read_text(encoding="utf-8"))
    attempt = json.loads(operations._submission_attempt_path(launch_path).read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert launch["artifact_type"] == "p7c4b2b_target_inner_preflight_launch_record"
    assert claim["artifact_type"] == "p7c4b2b_target_inner_preflight_submission_claim"
    assert attempt["artifact_type"] == "p7c4b2b_target_inner_preflight_submission_attempt"
    assert result["artifact_type"] == "p7c4b2b_target_inner_preflight_submission_result"
    assert receipt["schema_version"] == 2
    assert receipt["invocation_id"] == result["invocation_id"]
    assert receipt["unit_snapshot_status"] == "recovered_from_immutable_submission_result"
    assert result["stdout_sha256"] == hashlib.sha256(stdout_path.read_bytes()).hexdigest()
    assert result["stderr_sha256"] == hashlib.sha256(stderr_path.read_bytes()).hexdigest()
    assert result["exit_code_sha256"] == hashlib.sha256(exit_path.read_bytes()).hexdigest()


def test_b2b_cli_receipt_rejects_missing_durable_result(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    assert main([
        "create-submission-receipt", "--receipt", str(receipt_path),
        "--launch-record", str(launch_path), "--unit", launch["systemd_unit"],
        "--systemd-run-exit-code", "0",
    ]) == 2


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("target-inner-preflight", "p7c4b2b_target_inner_preflight_submission_result"),
        ("target-outer-projection-preflight", "p7c4b2c_target_outer_projection_preflight_submission_result"),
    ],
)
def test_submission_artifact_type_is_closed_and_stage_exact(stage, expected):
    assert operations._submission_artifact_type(stage, "result") == expected
    with pytest.raises(OperationsError, match="^launch_record_state_invalid$"):
        operations._submission_artifact_type("unsupported-stage", "result")


def test_b2b_receipt_rejects_outer_result_artifact_family(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)
    result["artifact_type"] = "p7c4b2c_target_outer_projection_preflight_submission_result"
    result["result_digest"] = operations._record_digest(result, "result_digest")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=None,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


def test_b2b_receipt_fails_closed_for_cross_invocation_snapshot(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, _result = _outer_submission_result(tmp_path, launch_path, launch)
    # The helper above creates the same durable evidence; only the live identity is adversarial.
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    snapshot.update({"LoadState": "loaded", "InvocationID": "1" * 32})
    with pytest.raises(OperationsError, match="^unit_snapshot_invocation_id_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path, launch_record_path=launch_path,
            unit_snapshot=snapshot, systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"], submission_result_path=result_path,
        )


def test_b2b_result_cannot_cross_bind_or_recover_another_invocation(tmp_path, monkeypatch):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first_path, first_receipt, first, _profile = _typed_inner_launch(
        tmp_path / "first", run_id="first", working_directory="/srv/repo-a",
        python_executable="/srv/repo-a/.venv/bin/python",
        output_directory="/srv/repo-a/artifacts/p7c4b2b-compute-preflight/first",
    )
    second_path, second_receipt, second, _profile = _typed_inner_launch(
        tmp_path / "second", run_id="second", working_directory="/srv/repo-b",
        python_executable="/srv/repo-b/.venv/bin/python",
        output_directory="/srv/repo-b/artifacts/p7c4b2b-compute-preflight/second",
    )
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: "b" * 40)
    create_submission_claim(launch_record_path=first_path, receipt_path=first_receipt)
    create_submission_claim(launch_record_path=second_path, receipt_path=second_receipt)
    first_result_path, first_result = _outer_submission_result(
        tmp_path / "first", first_path, first
    )
    second_result_path, second_result = _outer_submission_result(
        tmp_path / "second", second_path, second
    )
    assert first_result["invocation_id"] == second_result["invocation_id"]
    # Identical parsed IDs are still scoped to their immutable launch/claim/attempt
    # chain; a result from A cannot be used to recover B.
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=second_receipt,
            launch_record_path=second_path,
            unit_snapshot=None,
            systemd_run_exit_code=0,
            observed_unit=second["systemd_unit"],
            submission_result_path=first_result_path,
        )
    receipt = create_submission_receipt(
        receipt_path=second_receipt,
        launch_record_path=second_path,
        unit_snapshot=None,
        systemd_run_exit_code=0,
        observed_unit=second["systemd_unit"],
        submission_result_path=second_result_path,
    )
    assert receipt["submission_result_digest"] == second_result["result_digest"]


def test_b2b_recovery_fails_closed_when_immutable_output_is_tampered(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)
    Path(result["stderr_path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=None,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


def test_b2b_crash_after_systemd_submission_blocks_duplicate_invocation(tmp_path, monkeypatch):
    launch_path, receipt_path, launch, _profile = _typed_inner_launch(tmp_path)
    monkeypatch.setattr(operations, "_working_tree_git_head", lambda _path: launch["source_git_commit"])
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    monkeypatch.setattr(
        operations.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0,
            stdout=(f"Running as unit: {launch['systemd_unit']}; invocation ID: "
                    "09c753f133ac4a3fae89ba13ec21b3fe\n").encode(),
            stderr=b"",
        ),
    )
    monkeypatch.setattr(
        operations, "_atomic_bytes_create",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated crash")),
    )
    with pytest.raises(OSError, match="simulated crash"):
        operations.submit_systemd_run(
            result_path=tmp_path / "result.json",
            launch_record_path=launch_path,
            stdout_path=tmp_path / "stdout",
            stderr_path=tmp_path / "stderr",
            exit_code_path=tmp_path / "exit",
        )
    with pytest.raises(OperationsError, match="^submission_already_attempted$"):
        operations.submit_systemd_run(
            result_path=tmp_path / "retry-result.json",
            launch_record_path=launch_path,
            stdout_path=tmp_path / "retry-stdout",
            stderr_path=tmp_path / "retry-stderr",
            exit_code_path=tmp_path / "retry-exit",
        )


def test_outer_submit_attempt_is_one_winner_and_blocks_retry_after_capture_crash(
    tmp_path, monkeypatch
):
    launch_path, receipt_path, _launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    original_run = subprocess.run
    systemd_calls = 0

    def failed_run(command, **kwargs):
        nonlocal systemd_calls
        if command[0] == "git":
            return original_run(command, **kwargs)
        systemd_calls += 1
        raise OSError("simulated capture crash")

    monkeypatch.setattr(operations.subprocess, "run", failed_run)
    call = {
        "result_path": tmp_path / "submission-result.json",
        "launch_record_path": launch_path,
        "stdout_path": tmp_path / "systemd-run.stdout",
        "stderr_path": tmp_path / "systemd-run.stderr",
        "exit_code_path": tmp_path / "systemd-run.exit-code",
    }
    with pytest.raises(OperationsError, match="^systemd_run_capture_failed$"):
        operations.submit_systemd_run(**call)
    with pytest.raises(OperationsError, match="^submission_already_attempted$"):
        operations.submit_systemd_run(**call)
    assert systemd_calls == 1

    second_path, second_receipt, _second, _argv = _typed_outer_launch(
        tmp_path / "concurrent"
    )
    create_submission_claim(launch_record_path=second_path, receipt_path=second_receipt)
    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = [
            executor.submit(operations._create_submission_attempt, second_path)
            for _ in range(8)
        ]
    outcomes = []
    for attempt in attempts:
        try:
            outcomes.append(attempt.result()["attempt_state"])
        except OperationsError as exc:
            outcomes.append(str(exc))
    assert outcomes.count("submission_invocation_committed") == 1
    assert outcomes.count("submission_already_attempted") == 7


def test_outer_resume_submit_rechecks_original_unit_inactive(tmp_path, monkeypatch):
    launch_path, receipt_path, _launch, _argv = _typed_outer_launch(
        tmp_path, runner_command="resume"
    )
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    original_run = subprocess.run
    observed = []

    def active_unit(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        observed.append(command)
        if command[0] == "systemctl":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"LoadState=loaded\nActiveState=active\nMainPID=4242\n",
                stderr=b"",
            )
        raise AssertionError("systemd-run must not be invoked for an active original")

    monkeypatch.setattr(operations.subprocess, "run", active_unit)
    with pytest.raises(OperationsError, match="^resume_precheck_unit_active$"):
        operations.submit_systemd_run(
            result_path=tmp_path / "submission-result.json",
            launch_record_path=launch_path,
            stdout_path=tmp_path / "systemd-run.stdout",
            stderr_path=tmp_path / "systemd-run.stderr",
            exit_code_path=tmp_path / "systemd-run.exit-code",
        )
    assert len(observed) == 1
    assert observed[0][0] == "systemctl"
    assert not operations._submission_attempt_path(launch_path).exists()

    launch = json.loads(launch_path.read_text(encoding="utf-8"))

    def inactive_unit(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        if command[0] == "systemctl":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"LoadState=loaded\nActiveState=inactive\nMainPID=0\n",
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                f"Running as unit: {launch['systemd_unit']}; invocation ID: "
                "09c753f133ac4a3fae89ba13ec21b3fe\n"
            ).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(operations.subprocess, "run", inactive_unit)
    result = operations.submit_systemd_run(
        result_path=tmp_path / "submission-result.json",
        launch_record_path=launch_path,
        stdout_path=tmp_path / "systemd-run.stdout",
        stderr_path=tmp_path / "systemd-run.stderr",
        exit_code_path=tmp_path / "systemd-run.exit-code",
    )
    assert result["submission_state"] == "submitted"


@pytest.mark.parametrize("mutation", ["argv", "unit", "working", "log"])
def test_outer_claim_revalidates_self_redigested_launch_semantics(tmp_path, mutation):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    if mutation == "argv":
        launch["argv"].append("--unreviewed-scope")
    elif mutation == "unit":
        launch["systemd_unit"] = "unbound.service"
    elif mutation == "working":
        launch["working_directory"] = str(tmp_path / "different-repo")
        launch["resolved_working_directory"] = str(
            (tmp_path / "different-repo").resolve()
        )
    else:
        launch["log_path"] = str(tmp_path / "unbound.log")
        launch["resolved_log_path"] = str((tmp_path / "unbound.log").resolve())
    launch["record_digest"] = operations._record_digest(launch, "record_digest")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    with pytest.raises(OperationsError):
        create_submission_claim(
            launch_record_path=launch_path, receipt_path=receipt_path
        )
    assert not operations._submission_claim_path(launch_path).exists()


def test_outer_receipt_revalidates_self_redigested_attempt_semantics(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)
    attempt_path = Path(result["submission_attempt_path"])
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["artifact_type"] = "wrong_submission_attempt"
    attempt["attempt_digest"] = operations._record_digest(attempt, "attempt_digest")
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
    result["submission_attempt_sha256"] = hashlib.sha256(
        attempt_path.read_bytes()
    ).hexdigest()
    result["submission_attempt_digest"] = attempt["attempt_digest"]
    result["result_digest"] = operations._record_digest(result, "result_digest")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    snapshot["LoadState"] = "not-found"
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=snapshot,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


@pytest.mark.parametrize(
    "stdout,code",
    [
        ("Running as unit: UNIT;\n", "systemd_run_invocation_id_count_invalid"),
        (
            "Running as unit: UNIT; invocation ID: NOT-HEX\n",
            "invocation_id_malformed",
        ),
        (
            "Running as unit: UNIT; invocation ID: 09c753f133ac4a3fae89ba13ec21b3fe\n"
            "Running as unit: UNIT; invocation ID: 19c753f133ac4a3fae89ba13ec21b3fe\n",
            "systemd_run_invocation_id_count_invalid",
        ),
        (
            "Running as unit: WRONG; invocation ID: 09c753f133ac4a3fae89ba13ec21b3fe\n",
            "systemd_unit_mismatch",
        ),
    ],
)
def test_outer_submission_output_parser_is_closed(tmp_path, stdout, code):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    rendered = stdout.replace("UNIT", launch["systemd_unit"])
    with pytest.raises(OperationsError, match=f"^{code}$"):
        _outer_submission_result(tmp_path, launch_path, launch, stdout=rendered)


def test_outer_nonzero_exit_rejects_created_unit_or_invocation_evidence(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    successful_output = (
        f"Running as unit: {launch['systemd_unit']}; invocation ID: "
        "09c753f133ac4a3fae89ba13ec21b3fe\n"
    )
    with pytest.raises(OperationsError, match="^systemd_run_exit_code_mismatch$"):
        _outer_submission_result(
            tmp_path,
            launch_path,
            launch,
            exit_code=1,
            stdout=successful_output,
        )

    stderr_path, stderr_receipt, stderr_launch, _argv = _typed_outer_launch(
        tmp_path / "stderr-evidence"
    )
    create_submission_claim(
        launch_record_path=stderr_path, receipt_path=stderr_receipt
    )
    with pytest.raises(OperationsError, match="^systemd_run_exit_code_mismatch$"):
        _outer_submission_result(
            tmp_path / "stderr-evidence", stderr_path, stderr_launch,
            exit_code=1, stdout="", stderr=(
                f"Running as unit: {stderr_launch['systemd_unit']}; invocation ID: "
                "09c753f133ac4a3fae89ba13ec21b3fe\n"
            ),
        )

    second_path, second_receipt, second, _argv = _typed_outer_launch(tmp_path / "live")
    create_submission_claim(launch_record_path=second_path, receipt_path=second_receipt)
    result_path, _result = _outer_submission_result(
        tmp_path / "live", second_path, second, exit_code=1
    )
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    snapshot["LoadState"] = "loaded"
    snapshot["InvocationID"] = "09c753f133ac4a3fae89ba13ec21b3fe"
    with pytest.raises(OperationsError, match="^systemd_run_exit_code_mismatch$"):
        create_submission_receipt(
            receipt_path=second_receipt,
            launch_record_path=second_path,
            unit_snapshot=snapshot,
            systemd_run_exit_code=1,
            observed_unit=second["systemd_unit"],
            submission_result_path=result_path,
        )


def test_outer_receipt_rejects_altered_output_exit_and_live_identity(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)
    Path(result["stdout_path"]).write_text("altered\n", encoding="utf-8")
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=None,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )

    exit_path, exit_receipt, exit_launch, _argv = _typed_outer_launch(
        tmp_path / "exit-file"
    )
    create_submission_claim(launch_record_path=exit_path, receipt_path=exit_receipt)
    exit_result_path, exit_result = _outer_submission_result(
        tmp_path / "exit-file", exit_path, exit_launch
    )
    Path(exit_result["exit_code_path"]).write_bytes(b"1\n")
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=exit_receipt,
            launch_record_path=exit_path,
            unit_snapshot=None,
            systemd_run_exit_code=0,
            observed_unit=exit_launch["systemd_unit"],
            submission_result_path=exit_result_path,
        )

    second_path, second_receipt, second, _argv = _typed_outer_launch(tmp_path / "exit")
    create_submission_claim(launch_record_path=second_path, receipt_path=second_receipt)
    second_result_path, second_result = _outer_submission_result(
        tmp_path / "exit", second_path, second
    )
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=second_receipt,
            launch_record_path=second_path,
            unit_snapshot=None,
            systemd_run_exit_code=1,
            observed_unit=second["systemd_unit"],
            submission_result_path=second_result_path,
        )
    mismatch = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    mismatch["LoadState"] = "loaded"
    mismatch["InvocationID"] = "1" * 32
    with pytest.raises(OperationsError, match="^unit_snapshot_invocation_id_mismatch$"):
        create_submission_receipt(
            receipt_path=second_receipt,
            launch_record_path=second_path,
            unit_snapshot=mismatch,
            systemd_run_exit_code=0,
            observed_unit=second["systemd_unit"],
            submission_result_path=second_result_path,
        )


def test_outer_empty_snapshot_and_concurrent_recovery_are_no_clobber(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, _result = _outer_submission_result(tmp_path, launch_path, launch)
    empty_snapshot = tmp_path / "snapshot.json"
    empty_snapshot.write_bytes(b"")

    def recover():
        return create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=None,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
            snapshot_attempt_path=empty_snapshot,
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(recover) for _ in range(6)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result()["unit_snapshot_status"])
        except OperationsError as exc:
            outcomes.append(str(exc))
    assert outcomes.count("unavailable_empty_attempt") == 1
    assert outcomes.count("operational_evidence_collision") == 5


def test_outer_receipt_requires_live_or_immutable_snapshot_attempt(tmp_path):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, _result = _outer_submission_result(tmp_path, launch_path, launch)
    with pytest.raises(OperationsError, match="^unit_snapshot_evidence_missing$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=None,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_type", "wrong_claim"),
        ("execution_stage", "target-inner-preflight"),
        ("launch_record_path", "/wrong/launch.json"),
        ("submission_state", "submitted"),
    ],
)
def test_outer_submission_result_rejects_redigested_claim_semantic_mutation(
    tmp_path, field, value
):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    claim_path = operations._submission_claim_path(launch_path)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim[field] = value
    claim["claim_digest"] = operations._record_digest(claim, "claim_digest")
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(OperationsError, match="^submission_claim_invalid$"):
        _outer_submission_result(tmp_path, launch_path, launch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_type", "wrong_result"),
        ("submission_claim_path", "/wrong/claim.json"),
        ("submission_state", "submission_failed"),
    ],
)
def test_outer_receipt_rejects_redigested_result_semantic_mutation(
    tmp_path, field, value
):
    launch_path, receipt_path, launch, _argv = _typed_outer_launch(tmp_path)
    create_submission_claim(launch_record_path=launch_path, receipt_path=receipt_path)
    result_path, result = _outer_submission_result(tmp_path, launch_path, launch)
    result[field] = value
    result["result_digest"] = operations._record_digest(result, "result_digest")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    snapshot = {item: "" for item in UNIT_SNAPSHOT_FIELDS}
    snapshot["LoadState"] = "not-found"
    with pytest.raises(OperationsError, match="^submission_result_mismatch$"):
        create_submission_receipt(
            receipt_path=receipt_path,
            launch_record_path=launch_path,
            unit_snapshot=snapshot,
            systemd_run_exit_code=0,
            observed_unit=launch["systemd_unit"],
            submission_result_path=result_path,
        )


def test_typed_outer_projection_run_rejects_existing_output_and_mode_unit_cross_use(
    tmp_path,
):
    repo = tmp_path / "repo"
    output = repo / "artifacts" / "outer-run"
    output.mkdir(parents=True)
    with pytest.raises(OperationsError, match="^output_collision$"):
        _typed_outer_launch(
            tmp_path,
            working_directory=str(repo),
            output_directory=str(output),
        )
    with pytest.raises(OperationsError, match="^systemd_unit_mismatch$"):
        _typed_outer_launch(tmp_path / "cross", unit="p7c4b2c-outer-p2-wrong.service")


def test_typed_outer_projection_accepts_logical_repository_alias(tmp_path):
    physical = tmp_path / "physical"
    (physical / ".venv" / "bin").mkdir(parents=True)
    (physical / "artifacts").mkdir()
    logical = tmp_path / "logical"
    _directory_alias(logical, physical)
    logical_output = logical / "artifacts" / "outer-run"
    physical_output = physical / "artifacts" / "outer-run"
    _path, _receipt, launch, _argv = _typed_outer_launch(
        tmp_path,
        working_directory=str(logical),
        output_directory=str(logical_output),
        authorized_output_directory=str(physical_output),
    )
    assert launch["output_directory"] == str(logical_output)
    assert launch["resolved_output_directory"] == str(physical_output.resolve())


@pytest.mark.parametrize("case", ["different-physical", "traversal", "escape"])
def test_typed_outer_projection_rejects_invalid_output_identity(tmp_path, case):
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / "artifacts").mkdir()
    output = repo / "artifacts" / "outer-run"
    authorized = output
    if case == "different-physical":
        authorized = tmp_path / "other" / "outer-run"
    elif case == "traversal":
        output = repo / "artifacts" / ".." / "outside" / "outer-run"
        authorized = output
    else:
        outside = tmp_path / "outside" / "outer-run"
        outside.mkdir(parents=True)
        _directory_alias(output, outside)
        authorized = outside
    with pytest.raises(
        OperationsError,
        match="^(operational_identity_mismatch|operational_working_directory_mismatch)$",
    ):
        _typed_outer_launch(
            tmp_path,
            working_directory=str(repo),
            output_directory=str(output),
            authorized_output_directory=str(authorized),
        )


def test_outer_resume_precheck_rejects_active_unit_before_mutation(
    tmp_path, monkeypatch
):
    for name in (
        "plan.json",
        "manifest.json",
        "environment.json",
        "authorization_runtime.json",
    ):
        _input(tmp_path / name, "{}")
    monkeypatch.setattr(
        "creditrep.experiments.p7c4b2c_preflight.validate_artifacts",
        lambda _path: {"completed": 1, "expected": 2, "reason_codes": []},
    )
    snapshot = {field: "" for field in UNIT_SNAPSHOT_FIELDS}
    snapshot["ActiveState"] = "active"
    result = resume_precheck(tmp_path, unit_snapshot=snapshot)
    assert result["valid"] is False
    assert "resume_precheck_unit_active" in result["reason_codes"]
