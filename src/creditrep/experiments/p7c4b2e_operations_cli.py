"""Typed atomic operational evidence for target canary and inner preflight.

This module never creates an authorization, starts a runner, or reads datasets.
It writes immutable operator-side launch/submission evidence and performs a
read-only resume precheck over existing target-run artifacts. Artifact types
come only from the closed execution-stage mapping below.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from creditrep.strict_json import StrictJSONError, loads_strict_object


LAUNCH_RECORD_SCHEMA_VERSION = 1
SUBMISSION_RECEIPT_SCHEMA_VERSION = 1
SUBMISSION_CLAIM_SCHEMA_VERSION = 1
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
EXECUTION_ARTIFACT_TYPES = {
    "target-canary": (
        "p7c4b2e_target_canary_launch_record",
        "p7c4b2e_target_canary_submission_receipt",
    ),
    "target-inner-preflight": (
        "p7c4b2b_target_inner_preflight_launch_record",
        "p7c4b2b_target_inner_preflight_submission_receipt",
    ),
}
AUTHORIZATION_STAGE_BY_OPERATION_STAGE = {
    "target-inner-preflight": "target_inner_fit_projection_preflight",
}


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


def _resolve_operational_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise OperationsError("operational_identity_mismatch") from exc


def _resolved_operational_paths(paths: dict[str, Path]) -> dict[str, Path]:
    resolved = {name: _resolve_operational_path(path) for name, path in paths.items()}
    if any(
        _resolve_operational_path(paths[name]) != identity
        for name, identity in resolved.items()
    ):
        raise OperationsError("operational_identity_mismatch")
    return resolved


def _revalidate_operational_paths(
    paths: dict[str, Path], identities: dict[str, Path]
) -> None:
    if _resolved_operational_paths(paths) != identities:
        raise OperationsError("operational_identity_mismatch")


def _is_absolute_operational_path(value: str) -> bool:
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _record_digest(value: dict[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _typed_input(path: Path, digest_field: str) -> dict[str, Any]:
    value, source_hash = _json_input(path)
    artifact_digest = value.get(digest_field)
    if (
        not isinstance(artifact_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is None
        or artifact_digest != _record_digest(value, digest_field)
    ):
        raise OperationsError("control_file_digest_mismatch")
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": source_hash,
        "artifact_digest": artifact_digest,
    }


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise OperationsError("operational_evidence_collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
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
    *,
    record_path: Path,
    git_commit: str,
    operator_identity: str,
    authorization_path: Path,
    environment_path: Path,
    proposal_path: Path,
    unit: str,
    argv: list[Any],
    working_directory: str,
    python_executable: str,
    output_directory: str,
    log_path: str,
    execution_stage: str = "target-canary",
    machine_profile_path: Path | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise OperationsError("invalid_argv")
    if not isinstance(git_commit, str) or not GIT_COMMIT_RE.fullmatch(git_commit):
        raise OperationsError("invalid_git_commit")
    if execution_stage not in EXECUTION_ARTIFACT_TYPES:
        raise OperationsError("unknown_execution_stage")
    authorization, authorization_hash = _json_input(authorization_path)
    environment, environment_hash = _json_input(environment_path)
    proposal, proposal_hash = _json_input(proposal_path)
    artifact_type = EXECUTION_ARTIFACT_TYPES[execution_stage][0]
    if execution_stage == "target-inner-preflight":
        if machine_profile_path is None:
            raise OperationsError("machine_profile_missing")
        profile, _profile_hash = _json_input(machine_profile_path)
        authorized_output = authorization.get("normalized_output_directory")
        if (
            authorization.get("execution_stage")
            != AUTHORIZATION_STAGE_BY_OPERATION_STAGE[execution_stage]
            or environment.get("execution_stage")
            != AUTHORIZATION_STAGE_BY_OPERATION_STAGE[execution_stage]
            or proposal.get("execution_stage")
            != AUTHORIZATION_STAGE_BY_OPERATION_STAGE[execution_stage]
            or authorization.get("source_git_commit") != git_commit
            or environment.get("source_git_commit") != git_commit
            or not isinstance(authorized_output, str)
            or not authorized_output
            or profile.get("git_commit") != git_commit
            or profile.get("profile_digest")
            != authorization.get("machine_profile_digest")
        ):
            raise OperationsError("operational_identity_mismatch")
        working_path = Path(working_directory)
        python_path = Path(python_executable)
        output_path = Path(output_directory)
        authorized_output_path = Path(authorized_output)
        expected_python = working_path / ".venv" / "bin" / "python"
        expected_output_root = working_path / "artifacts" / "p7c4b2b-compute-preflight"
        if (
            not all(
                _is_absolute_operational_path(value)
                for value in (
                    working_directory,
                    python_executable,
                    output_directory,
                    authorized_output,
                )
            )
            or any(
                ".." in path.parts
                for path in (
                    working_path,
                    python_path,
                    output_path,
                    authorized_output_path,
                )
            )
            or output_path.is_symlink()
        ):
            raise OperationsError("operational_working_directory_mismatch")
        operational_paths = {
            "working": working_path,
            "python": python_path,
            "expected_python": expected_python,
            "output": output_path,
            "authorized_output": authorized_output_path,
            "output_root": expected_output_root,
        }
        resolved_paths = _resolved_operational_paths(operational_paths)
        if resolved_paths["output"] != resolved_paths["authorized_output"]:
            raise OperationsError("operational_identity_mismatch")
        try:
            resolved_paths["output"].relative_to(resolved_paths["output_root"])
        except ValueError as exc:
            raise OperationsError("operational_working_directory_mismatch") from exc
        if resolved_paths["python"] != resolved_paths[
            "expected_python"
        ] or output_path.name != authorization.get("run_id"):
            raise OperationsError("operational_working_directory_mismatch")
        if (
            not unit.startswith("p7c4b2b-inner-")
            or authorization.get("run_id") not in unit
        ):
            raise OperationsError("systemd_unit_mismatch")
        runner_command = argv[3] if len(argv) >= 4 else None
        expected_argv = [
            python_executable,
            "-m",
            "creditrep.experiments.p7c4b2b_cli",
            runner_command,
            "--mode",
            authorization.get("mode"),
            "--profile",
            str(machine_profile_path),
            "--target-machine-asserted",
            "--target-environment",
            str(environment_path),
            "--authorization-proposal",
            str(proposal_path),
            "--effective-authorization",
            str(authorization_path),
            "--output-dir",
            output_directory,
        ]
        if runner_command not in {"run", "resume"} or argv != expected_argv:
            raise OperationsError("operational_argv_mismatch")
        _revalidate_operational_paths(operational_paths, resolved_paths)
    value = {
        "schema_version": LAUNCH_RECORD_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "source_git_commit": _nonempty(git_commit),
        "created_at": _utc_now(),
        "operator_identity": _nonempty(operator_identity),
        "authorization": {
            "path": str(authorization_path),
            "sha256": authorization_hash,
        },
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
    if execution_stage == "target-inner-preflight":
        value.update(
            {
                "execution_stage": execution_stage,
                "runner_command": runner_command,
                "run_id": authorization.get("run_id"),
                "machine_profile": _typed_input(machine_profile_path, "profile_digest"),
                "environment": _typed_input(environment_path, "environment_digest"),
                "proposal": _typed_input(proposal_path, "proposal_digest"),
                "authorization": _typed_input(
                    authorization_path, "authorization_digest"
                ),
            }
        )
        value["record_digest"] = _record_digest(value, "record_digest")
    _atomic_create(record_path, value)
    return value


def _submission_claim_path(launch_record_path: Path) -> Path:
    return launch_record_path.with_name(
        f".{launch_record_path.name}.submission-claim.json"
    )


def create_submission_claim(
    *, launch_record_path: Path, receipt_path: Path
) -> dict[str, Any]:
    record, launch_hash = _json_input(launch_record_path)
    stage = record.get("execution_stage")
    if (
        stage != "target-inner-preflight"
        or record.get("artifact_type") != EXECUTION_ARTIFACT_TYPES[stage][0]
        or record.get("schema_version") != LAUNCH_RECORD_SCHEMA_VERSION
        or record.get("submission_state") != "prepared_not_submitted"
        or record.get("record_digest") != _record_digest(record, "record_digest")
    ):
        raise OperationsError("launch_record_state_invalid")
    value = {
        "schema_version": SUBMISSION_CLAIM_SCHEMA_VERSION,
        "artifact_type": "p7c4b2b_target_inner_preflight_submission_claim",
        "execution_stage": stage,
        "claimed_at": _utc_now(),
        "launch_record_path": str(launch_record_path.resolve()),
        "launch_record_sha256": launch_hash,
        "receipt_path": str(receipt_path.resolve()),
        "systemd_unit": record["systemd_unit"],
        "submission_state": "claimed_not_submitted",
    }
    value["claim_digest"] = _record_digest(value, "claim_digest")
    try:
        _atomic_create(_submission_claim_path(launch_record_path), value)
    except OperationsError as exc:
        if str(exc) == "operational_evidence_collision":
            raise OperationsError("duplicate_submission") from exc
        raise
    return value


def create_submission_receipt(
    *,
    receipt_path: Path,
    launch_record_path: Path,
    unit_snapshot: dict[str, Any],
    systemd_run_exit_code: int,
    observed_unit: str | None = None,
) -> dict[str, Any]:
    if set(unit_snapshot) != UNIT_SNAPSHOT_FIELDS:
        raise OperationsError("unit_snapshot_schema_mismatch")
    if any(not isinstance(value, str) for value in unit_snapshot.values()):
        raise OperationsError("unit_snapshot_schema_mismatch")
    if not isinstance(systemd_run_exit_code, int) or isinstance(
        systemd_run_exit_code, bool
    ):
        raise OperationsError("invalid_systemd_run_exit_code")
    record, launch_hash = _json_input(launch_record_path)
    stage = record.get("execution_stage", "target-canary")
    if stage not in EXECUTION_ARTIFACT_TYPES or (
        record.get("artifact_type") != EXECUTION_ARTIFACT_TYPES[stage][0]
        or record.get("schema_version") != LAUNCH_RECORD_SCHEMA_VERSION
        or not isinstance(record.get("systemd_unit"), str)
        or record.get("submission_state") != "prepared_not_submitted"
    ):
        raise OperationsError("launch_record_state_invalid")
    value = {
        "schema_version": SUBMISSION_RECEIPT_SCHEMA_VERSION,
        "artifact_type": EXECUTION_ARTIFACT_TYPES[stage][1],
        "created_at": _utc_now(),
        "launch_record_path": str(launch_record_path),
        "launch_record_sha256": launch_hash,
        "systemd_unit": record["systemd_unit"],
        "systemd_run_exit_code": systemd_run_exit_code,
        "unit_snapshot": unit_snapshot,
        "submission_state": "systemd_run_returned",
    }
    if stage == "target-inner-preflight":
        if record.get("record_digest") != _record_digest(record, "record_digest"):
            raise OperationsError("launch_record_digest_mismatch")
        invocation_id = unit_snapshot.get("InvocationID")
        if observed_unit != record.get("systemd_unit"):
            raise OperationsError("systemd_unit_mismatch")
        if systemd_run_exit_code == 0 and not invocation_id:
            raise OperationsError("invocation_id_missing")
        claim_path = _submission_claim_path(launch_record_path)
        claim, claim_hash = _json_input(claim_path)
        if claim.get("receipt_path") != str(receipt_path.resolve()):
            raise OperationsError("duplicate_submission")
        if (
            claim.get("schema_version") != SUBMISSION_CLAIM_SCHEMA_VERSION
            or claim.get("artifact_type")
            != "p7c4b2b_target_inner_preflight_submission_claim"
            or claim.get("execution_stage") != stage
            or claim.get("launch_record_path") != str(launch_record_path.resolve())
            or claim.get("launch_record_sha256") != launch_hash
            or claim.get("systemd_unit") != record["systemd_unit"]
            or claim.get("submission_state") != "claimed_not_submitted"
            or claim.get("claim_digest") != _record_digest(claim, "claim_digest")
        ):
            raise OperationsError("submission_claim_invalid")
        value.update(
            {
                "execution_stage": stage,
                "invocation_id": invocation_id,
                "submission_claim_path": str(claim_path.resolve()),
                "submission_claim_sha256": claim_hash,
                "submission_claim_digest": claim["claim_digest"],
                "submitted_at": value.pop("created_at"),
                "submission_state": "submitted"
                if systemd_run_exit_code == 0
                else "submission_failed",
            }
        )
        value["receipt_digest"] = _record_digest(value, "receipt_digest")
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
        "reason_codes": sorted(
            set(
                [
                    *report.get("reason_codes", []),
                    *(["resume_precheck_missing_required_artifact"] if missing else []),
                    *(["resume_precheck_not_incomplete"] if not incomplete else []),
                    *(
                        ["resume_precheck_completed_marker_present"]
                        if (run_dir / "COMPLETED.json").exists()
                        else []
                    ),
                ]
            )
        ),
        "completed": completed,
        "expected": expected,
        "resume_requires_existing_authorization_revalidation": True,
    }


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "create-launch-record",
            "create-submission-claim",
            "create-submission-receipt",
            "resume-precheck",
        ),
    )
    parser.add_argument("--record", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--operator-identity")
    parser.add_argument(
        "--execution-stage",
        default="target-canary",
        choices=tuple(EXECUTION_ARTIFACT_TYPES),
    )
    parser.add_argument("--machine-profile", type=Path)
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
            required = (
                args.record,
                args.git_commit,
                args.operator_identity,
                args.authorization,
                args.environment,
                args.proposal,
                args.unit,
                args.argv_json,
                args.working_directory,
                args.python_executable,
                args.output_directory,
                args.log_path,
            )
            if any(item is None for item in required):
                raise OperationsError("missing_required_operational_argument")
            value = create_launch_record(
                record_path=args.record,
                git_commit=args.git_commit,
                operator_identity=args.operator_identity,
                authorization_path=args.authorization,
                environment_path=args.environment,
                proposal_path=args.proposal,
                unit=args.unit,
                argv=json.loads(args.argv_json),
                working_directory=args.working_directory,
                python_executable=args.python_executable,
                output_directory=args.output_directory,
                log_path=args.log_path,
                execution_stage=args.execution_stage,
                machine_profile_path=args.machine_profile,
            )
        elif args.command == "create-submission-claim":
            if args.launch_record is None or args.receipt is None:
                raise OperationsError("missing_required_operational_argument")
            value = create_submission_claim(
                launch_record_path=args.launch_record,
                receipt_path=args.receipt,
            )
        elif args.command == "create-submission-receipt":
            if None in (
                args.receipt,
                args.launch_record,
                args.unit_snapshot_json,
                args.systemd_run_exit_code,
            ):
                raise OperationsError("missing_required_operational_argument")
            value = create_submission_receipt(
                receipt_path=args.receipt,
                launch_record_path=args.launch_record,
                unit_snapshot=loads_strict_object(args.unit_snapshot_json),
                systemd_run_exit_code=args.systemd_run_exit_code,
                observed_unit=args.unit,
            )
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
