"""Atomic operational evidence helpers for the P7C.4B.2d target canary.

This module never creates an authorization, starts a runner, or reads datasets.
It writes immutable operator-side launch evidence and performs a read-only resume
precheck over the existing target-run artifacts.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from creditrep.strict_json import StrictJSONError, loads_strict_object


LAUNCH_RECORD_SCHEMA_VERSION = 1
SUBMISSION_RECEIPT_SCHEMA_VERSION = 1
UNIT_SNAPSHOT_FIELDS = frozenset(
    {
        "LoadState",
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Result",
        "InvocationID",
        "ExecMainStartTimestamp",
        "ExecMainExitTimestamp",
    }
)
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class OperationsError(ValueError):
    """Stable operational-evidence failure."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise OperationsError("evidence_input_missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_input(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = loads_strict_object(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OperationsError("evidence_input_missing") from exc
    except (OSError, UnicodeDecodeError, StrictJSONError) as exc:
        raise OperationsError("evidence_input_invalid") from exc
    return value, _sha256_file(path)


def _nonempty(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OperationsError("invalid_operational_value")
    return value.strip()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise OperationsError("operational_evidence_collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise OperationsError("operational_evidence_collision") from exc
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    except OSError as exc:
        Path(temporary_name).unlink(missing_ok=True)
        if isinstance(exc, OperationsError):
            raise
        raise OperationsError("operational_evidence_write_failed") from exc


def create_launch_record(
    *, record_path: Path, git_commit: str, operator_identity: str,
    authorization_path: Path, environment_path: Path, proposal_path: Path,
    unit: str, argv: list[Any], working_directory: str, python_executable: str,
    output_directory: str, log_path: str,
) -> dict[str, Any]:
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise OperationsError("invalid_argv")
    if not isinstance(git_commit, str) or not GIT_COMMIT_RE.fullmatch(git_commit):
        raise OperationsError("invalid_git_commit")
    _authorization, authorization_hash = _json_input(authorization_path)
    _environment, environment_hash = _json_input(environment_path)
    _proposal, proposal_hash = _json_input(proposal_path)
    value = {
        "schema_version": LAUNCH_RECORD_SCHEMA_VERSION,
        "artifact_type": "p7c4b2e_target_canary_launch_record",
        "source_git_commit": _nonempty(git_commit),
        "created_at": _utc_now(),
        "operator_identity": _nonempty(operator_identity),
        "authorization": {"path": str(authorization_path), "sha256": authorization_hash},
        "environment": {"path": str(environment_path), "sha256": environment_hash},
        "proposal": {"path": str(proposal_path), "sha256": proposal_hash},
        "systemd_unit": _nonempty(unit),
        "argv": argv,
        "working_directory": _nonempty(working_directory),
        "python_executable": _nonempty(python_executable),
        "output_directory": _nonempty(output_directory),
        "log_path": _nonempty(log_path),
        "execution_class": "target_preflight",
        "submission_state": "prepared_not_submitted",
    }
    _atomic_create(record_path, value)
    return value


def create_submission_receipt(
    *, receipt_path: Path, launch_record_path: Path, unit_snapshot: dict[str, Any],
    systemd_run_exit_code: int,
) -> dict[str, Any]:
    if set(unit_snapshot) != UNIT_SNAPSHOT_FIELDS:
        raise OperationsError("unit_snapshot_schema_mismatch")
    if any(not isinstance(value, str) for value in unit_snapshot.values()):
        raise OperationsError("unit_snapshot_schema_mismatch")
    if not isinstance(systemd_run_exit_code, int) or isinstance(systemd_run_exit_code, bool):
        raise OperationsError("invalid_systemd_run_exit_code")
    record, launch_hash = _json_input(launch_record_path)
    if (
        record.get("artifact_type") != "p7c4b2e_target_canary_launch_record"
        or record.get("schema_version") != LAUNCH_RECORD_SCHEMA_VERSION
        or not isinstance(record.get("systemd_unit"), str)
        or record.get("submission_state") != "prepared_not_submitted"
    ):
        raise OperationsError("launch_record_state_invalid")
    value = {
        "schema_version": SUBMISSION_RECEIPT_SCHEMA_VERSION,
        "artifact_type": "p7c4b2e_target_canary_submission_receipt",
        "created_at": _utc_now(),
        "launch_record_path": str(launch_record_path),
        "launch_record_sha256": launch_hash,
        "systemd_unit": record["systemd_unit"],
        "systemd_run_exit_code": systemd_run_exit_code,
        "unit_snapshot": unit_snapshot,
        "submission_state": "systemd_run_returned",
    }
    _atomic_create(receipt_path, value)
    return value


def resume_precheck(run_dir: Path) -> dict[str, Any]:
    required = (
        "plan.json",
        "manifest.json",
        "environment.json",
        "authorization_runtime.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return {
            "valid": False,
            "reason_codes": ["resume_precheck_missing_required_artifact"],
            "completed": None,
            "expected": None,
            "resume_requires_existing_authorization_revalidation": True,
        }
    # Import only after the cheap structural fail-closed checks. Launch-record
    # creation and malformed-run rejection must remain usable before the target
    # ML runtime is imported or validated.
    from creditrep.experiments.p7c4b2c_preflight import validate_artifacts

    report = validate_artifacts(run_dir)
    completed, expected = report.get("completed"), report.get("expected")
    incomplete = (
        isinstance(completed, int)
        and isinstance(expected, int)
        and 0 <= completed < expected
    )
    eligible = incomplete and not (run_dir / "COMPLETED.json").exists()
    return {
        "valid": eligible,
        "reason_codes": sorted(set([*report.get("reason_codes", []), *(
            ["resume_precheck_missing_required_artifact"] if missing else []
        ), *(
            ["resume_precheck_not_incomplete"] if not incomplete else []
        ), *(
            ["resume_precheck_completed_marker_present"]
            if (run_dir / "COMPLETED.json").exists()
            else []
        )])),
        "completed": completed,
        "expected": expected,
        "resume_requires_existing_authorization_revalidation": True,
    }


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create-launch-record", "create-submission-receipt", "resume-precheck"))
    parser.add_argument("--record", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--operator-identity")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--unit")
    parser.add_argument("--argv-json")
    parser.add_argument("--working-directory")
    parser.add_argument("--python-executable")
    parser.add_argument("--output-directory")
    parser.add_argument("--log-path")
    parser.add_argument("--launch-record", type=Path)
    parser.add_argument("--unit-snapshot-json")
    parser.add_argument("--systemd-run-exit-code", type=int)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "create-launch-record":
            required = (args.record, args.git_commit, args.operator_identity, args.authorization, args.environment, args.proposal, args.unit, args.argv_json, args.working_directory, args.python_executable, args.output_directory, args.log_path)
            if any(item is None for item in required):
                raise OperationsError("missing_required_operational_argument")
            value = create_launch_record(record_path=args.record, git_commit=args.git_commit, operator_identity=args.operator_identity, authorization_path=args.authorization, environment_path=args.environment, proposal_path=args.proposal, unit=args.unit, argv=json.loads(args.argv_json), working_directory=args.working_directory, python_executable=args.python_executable, output_directory=args.output_directory, log_path=args.log_path)
        elif args.command == "create-submission-receipt":
            if None in (args.receipt, args.launch_record, args.unit_snapshot_json, args.systemd_run_exit_code):
                raise OperationsError("missing_required_operational_argument")
            value = create_submission_receipt(receipt_path=args.receipt, launch_record_path=args.launch_record, unit_snapshot=loads_strict_object(args.unit_snapshot_json), systemd_run_exit_code=args.systemd_run_exit_code)
        else:
            if args.run_dir is None:
                raise OperationsError("missing_required_operational_argument")
            value = resume_precheck(args.run_dir)
        _print(value)
        return 0 if value.get("valid", True) else 3
    except (OperationsError, StrictJSONError, OSError, json.JSONDecodeError) as exc:
        _print({"valid": False, "reason_codes": [str(exc)]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
