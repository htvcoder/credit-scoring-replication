from __future__ import annotations

import hashlib
import json

import pytest

from creditrep.experiments.p7c4b2e_operations_cli import (
    UNIT_SNAPSHOT_FIELDS,
    create_launch_record,
    create_submission_receipt,
    resume_precheck,
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
        record_path=record, git_commit="a" * 40, operator_identity="operator",
        authorization_path=auth, environment_path=environment, proposal_path=proposal,
        unit="p7c4b2d-target", argv=["python", "-m", "creditrep"],
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
        receipt_path=receipt, launch_record_path=record, unit_snapshot=snapshot,
        systemd_run_exit_code=0,
    )
    assert value["launch_record_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert value["unit_snapshot"] == snapshot
    try:
        create_submission_receipt(
            receipt_path=receipt, launch_record_path=record, unit_snapshot=snapshot,
            systemd_run_exit_code=0,
        )
    except Exception as exc:
        assert str(exc) == "operational_evidence_collision"
    else:
        raise AssertionError("receipt was overwritten")


def test_helper_rejects_malformed_inputs_and_snapshot_schema(tmp_path):
    record, _value = _launch(tmp_path)
    bad = _input(tmp_path / "bad.json", "not json")
    with pytest.raises(Exception, match="evidence_input_invalid"):
        create_launch_record(
            record_path=tmp_path / "other.json", git_commit="a" * 40,
            operator_identity="operator", authorization_path=bad,
            environment_path=bad, proposal_path=bad, unit="unit", argv=["python"],
            working_directory="/srv", python_executable="/srv/python",
            output_directory="/srv/artifacts/out", log_path="/secure/log",
        )
    with pytest.raises(Exception, match="unit_snapshot_schema_mismatch"):
        create_submission_receipt(
            receipt_path=tmp_path / "bad-receipt.json", launch_record_path=record,
            unit_snapshot={}, systemd_run_exit_code=0,
        )


def test_atomic_write_failure_leaves_no_final_record(tmp_path, monkeypatch):
    import creditrep.experiments.p7c4b2e_operations_cli as operations

    def fail_link(*_args):
        raise OSError("link failed")

    monkeypatch.setattr(operations.os, "link", fail_link)
    auth = _input(tmp_path / "authorization.json", "{}")
    with pytest.raises(Exception, match="operational_evidence_write_failed"):
        create_launch_record(
            record_path=tmp_path / "launch.json", git_commit="a" * 40,
            operator_identity="operator", authorization_path=auth,
            environment_path=auth, proposal_path=auth, unit="unit", argv=["python"],
            working_directory="/srv", python_executable="/srv/python",
            output_directory="/srv/artifacts/out", log_path="/secure/log",
        )
    assert not (tmp_path / "launch.json").exists()
    assert not list(tmp_path.glob(".launch.json.*.tmp"))


def test_resume_precheck_fails_closed_when_required_run_artifacts_are_absent(tmp_path):
    value = resume_precheck(tmp_path)
    assert value["valid"] is False
    assert "resume_precheck_missing_required_artifact" in value["reason_codes"]
