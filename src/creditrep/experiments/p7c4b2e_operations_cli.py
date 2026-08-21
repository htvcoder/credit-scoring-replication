"""Typed atomic operational evidence for target preflight execution stages.

This module never creates an authorization or reads datasets. Its explicit
``submit-systemd-run`` command is the single typed compute boundary: it submits
the exact recorded argv and durably captures stdout, stderr, exit code and the
submission result before returning. Other commands write immutable operator-side
evidence or perform a read-only resume precheck. Artifact types come only from
the closed execution-stage mapping below.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import tempfile
from typing import Any

from creditrep.strict_json import StrictJSONError, loads_strict_object


LAUNCH_RECORD_SCHEMA_VERSION = 1
SUBMISSION_RECEIPT_SCHEMA_VERSION = 1
SUBMISSION_CLAIM_SCHEMA_VERSION = 1
OUTER_LAUNCH_RECORD_SCHEMA_VERSION = 2
OUTER_SUBMISSION_CLAIM_SCHEMA_VERSION = 2
OUTER_SUBMISSION_RESULT_SCHEMA_VERSION = 2
OUTER_SUBMISSION_RECEIPT_SCHEMA_VERSION = 2
B2B_SUBMISSION_RESULT_SCHEMA_VERSION = 2
B2B_SUBMISSION_RECEIPT_SCHEMA_VERSION = 2
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
    "target-outer-projection-preflight": (
        "p7c4b2c_target_outer_projection_preflight_launch_record",
        "p7c4b2c_target_outer_projection_preflight_submission_receipt",
    ),
}
AUTHORIZATION_STAGE_BY_OPERATION_STAGE = {
    "target-inner-preflight": "target_inner_fit_projection_preflight",
    "target-outer-projection-preflight": "target_projection_preflight",
}
SUBMISSION_CLAIM_ARTIFACT_TYPES = {
    "target-inner-preflight": "p7c4b2b_target_inner_preflight_submission_claim",
    "target-outer-projection-preflight": (
        "p7c4b2c_target_outer_projection_preflight_submission_claim"
    ),
}
SUBMISSION_STAGE_PREFIXES = {
    "target-inner-preflight": "p7c4b2b_target_inner_preflight",
    "target-outer-projection-preflight": (
        "p7c4b2c_target_outer_projection_preflight"
    ),
}
OUTER_MODES = frozenset({"cpu_parallel_1", "cpu_parallel_2"})
OUTER_DATASET_IDS = ("AC", "GC", "TH02", "HMEQ", "TC", "GMC")
OUTER_LAUNCH_RECORD_FIELDS = frozenset(
    {
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
        "execution_stage",
        "protocol_stage",
        "runner_command",
        "mode",
        "run_id",
        "resolved_working_directory",
        "resolved_python_executable",
        "resolved_output_directory",
        "resolved_log_path",
        "task_ids",
        "task_set_digest",
        "record_digest",
    }
)
OUTER_SUBMISSION_CLAIM_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "execution_stage",
        "claimed_at",
        "launch_record_path",
        "launch_record_sha256",
        "launch_record_digest",
        "receipt_path",
        "systemd_unit",
        "submission_state",
        "claim_digest",
    }
)
OUTER_SUBMISSION_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "created_at",
        "launch_record_path",
        "launch_record_sha256",
        "launch_record_digest",
        "submission_claim_path",
        "submission_claim_sha256",
        "submission_claim_digest",
        "submission_attempt_path",
        "submission_attempt_sha256",
        "submission_attempt_digest",
        "systemd_unit",
        "submission_capture_path",
        "submission_capture_sha256",
        "submission_capture_digest",
        "stdout_path",
        "stdout_sha256",
        "stderr_path",
        "stderr_sha256",
        "systemd_run_exit_code",
        "exit_code_path",
        "exit_code_sha256",
        "invocation_id",
        "submission_state",
        "result_digest",
    }
)
OUTER_SUBMISSION_ATTEMPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "attempted_at",
        "launch_record_path",
        "launch_record_sha256",
        "launch_record_digest",
        "submission_claim_path",
        "submission_claim_sha256",
        "submission_claim_digest",
        "systemd_unit",
        "attempt_state",
        "attempt_digest",
    }
)
SUBMISSION_CAPTURE_FIELDS = frozenset(
    {
        "schema_version", "artifact_type", "created_at", "execution_stage",
        "launch_record_path", "launch_record_sha256", "launch_record_digest",
        "submission_claim_path", "submission_claim_sha256", "submission_claim_digest",
        "submission_attempt_path", "submission_attempt_sha256", "submission_attempt_digest",
        "systemd_unit", "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
        "exit_code_path", "exit_code_sha256", "systemd_run_exit_code", "capture_digest",
    }
)
B2B_LAUNCH_RECORD_FIELDS = frozenset(
    {
        "schema_version", "artifact_type", "source_git_commit", "created_at",
        "operator_identity", "authorization", "environment", "proposal",
        "systemd_unit", "argv", "working_directory", "python_executable",
        "output_directory", "log_path", "execution_class", "submission_state",
        "execution_stage", "runner_command", "run_id", "machine_profile",
        "record_digest",
    }
)
B2B_SUBMISSION_CLAIM_FIELDS = frozenset(
    {
        "schema_version", "artifact_type", "execution_stage", "claimed_at",
        "launch_record_path", "launch_record_sha256", "receipt_path",
        "systemd_unit", "submission_state", "claim_digest",
    }
)
B2B_SUBMISSION_ATTEMPT_FIELDS = frozenset(
    {
        "schema_version", "artifact_type", "attempted_at", "launch_record_path",
        "launch_record_sha256", "launch_record_digest", "submission_claim_path",
        "submission_claim_sha256", "submission_claim_digest", "systemd_unit",
        "attempt_state", "attempt_digest",
    }
)
B2B_SUBMISSION_RESULT_FIELDS = frozenset(
    {
        "schema_version", "artifact_type", "created_at", "launch_record_path",
        "launch_record_sha256", "launch_record_digest", "submission_claim_path",
        "submission_claim_sha256", "submission_claim_digest", "submission_attempt_path",
        "submission_attempt_sha256", "submission_attempt_digest", "systemd_unit",
        "submission_capture_path", "submission_capture_sha256", "submission_capture_digest",
        "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
        "systemd_run_exit_code", "exit_code_path", "exit_code_sha256",
        "invocation_id", "submission_state", "result_digest",
    }
)
SUBMISSION_STAGE_POLICIES = {
    "target-inner-preflight": {
        "claim_fields": B2B_SUBMISSION_CLAIM_FIELDS,
        "claim_schema": SUBMISSION_CLAIM_SCHEMA_VERSION,
        "attempt_fields": B2B_SUBMISSION_ATTEMPT_FIELDS,
        "capture_fields": SUBMISSION_CAPTURE_FIELDS,
        "result_fields": B2B_SUBMISSION_RESULT_FIELDS,
        "result_schema": B2B_SUBMISSION_RESULT_SCHEMA_VERSION,
        "receipt_schema": B2B_SUBMISSION_RECEIPT_SCHEMA_VERSION,
    },
    "target-outer-projection-preflight": {
        "claim_fields": OUTER_SUBMISSION_CLAIM_FIELDS,
        "claim_schema": OUTER_SUBMISSION_CLAIM_SCHEMA_VERSION,
        "attempt_fields": OUTER_SUBMISSION_ATTEMPT_FIELDS,
        "capture_fields": SUBMISSION_CAPTURE_FIELDS,
        "result_fields": OUTER_SUBMISSION_RESULT_FIELDS,
        "result_schema": OUTER_SUBMISSION_RESULT_SCHEMA_VERSION,
        "receipt_schema": OUTER_SUBMISSION_RECEIPT_SCHEMA_VERSION,
    },
}
INVOCATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class OperationsError(ValueError):
    """Stable operational-evidence failure."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise OperationsError("evidence_input_missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_exit_code_evidence(path: Path) -> tuple[int, str]:
    try:
        payload = path.read_bytes()
        value_text = payload.decode("ascii")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        raise OperationsError("systemd_run_exit_code_evidence_invalid") from exc
    if re.fullmatch(r"(?:0|[1-9][0-9]{0,2})\n", value_text) is None:
        raise OperationsError("systemd_run_exit_code_evidence_invalid")
    value = int(value_text.strip())
    if not 0 <= value <= 255:
        raise OperationsError("systemd_run_exit_code_evidence_invalid")
    return value, hashlib.sha256(payload).hexdigest()


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


def _working_tree_git_head(working_directory: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(working_directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OperationsError("operational_git_provenance_mismatch") from exc
    head = result.stdout.strip()
    if result.returncode != 0 or GIT_COMMIT_RE.fullmatch(head) is None:
        raise OperationsError("operational_git_provenance_mismatch")
    return head


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


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _outer_control_identity(
    *,
    authorization: dict[str, Any],
    environment: dict[str, Any],
    proposal: dict[str, Any],
    git_commit: str,
) -> tuple[str, str, list[str]]:
    stage = AUTHORIZATION_STAGE_BY_OPERATION_STAGE["target-outer-projection-preflight"]
    mode = authorization.get("execution_mode")
    task_ids = authorization.get("task_ids")
    dataset_hashes = authorization.get("dataset_hashes")
    if (
        mode not in OUTER_MODES
        or not isinstance(task_ids, list)
        or len(task_ids) != 162
        or any(not isinstance(item, str) or not item for item in task_ids)
        or len(set(task_ids)) != 162
        or any(
            value.get("execution_stage") != stage for value in (proposal, authorization)
        )
        or any(
            value.get("git_commit") != git_commit
            for value in (environment, proposal, authorization)
        )
        or any(
            value.get("execution_mode") != mode
            for value in (environment, proposal, authorization)
        )
        or proposal.get("target_environment_digest")
        != environment.get("environment_digest")
        or authorization.get("target_environment_digest")
        != environment.get("environment_digest")
        or authorization.get("proposal_digest") != proposal.get("proposal_digest")
        or proposal.get("task_ids") != task_ids
        or any(
            value.get("dataset_ids") != list(OUTER_DATASET_IDS)
            for value in (environment, proposal, authorization)
        )
        or not isinstance(dataset_hashes, dict)
        # dataset_hashes is a semantic mapping.  CLI control files are
        # serialized with sort_keys=True, so JSON member order is not part of
        # this identity contract; dataset_ids remains ordered above.
        or set(dataset_hashes) != set(OUTER_DATASET_IDS)
        or any(
            not isinstance(item, str) or re.fullmatch(r"[A-Fa-f0-9]{64}", item) is None
            for item in dataset_hashes.values()
        )
        or any(
            value.get("dataset_hashes") != dataset_hashes
            for value in (environment, proposal)
        )
        or any(
            value.get("dataset_binding_digest")
            != authorization.get("dataset_binding_digest")
            for value in (environment, proposal)
        )
        or authorization.get("maximum_task_count") != 162
        or proposal.get("maximum_task_count") != 162
    ):
        raise OperationsError("operational_identity_mismatch")
    return mode, _canonical_digest(task_ids), task_ids


def _outer_expected_argv(
    *,
    runner_command: str,
    python_executable: str,
    mode: str,
    output_directory: str,
    environment_path: Path,
    proposal_path: Path,
    authorization_path: Path,
) -> list[str]:
    prefix = [
        python_executable,
        "-m",
        "creditrep.experiments.p7c4b2c_cli",
        runner_command,
    ]
    controls = [
        "--target-environment",
        str(environment_path),
        "--authorization-proposal",
        str(proposal_path),
        "--effective-authorization",
        str(authorization_path),
    ]
    if runner_command == "run":
        return [
            *prefix,
            "--execution-class",
            "target_preflight",
            "--mode",
            mode,
            "--output",
            output_directory,
            *controls,
        ]
    if runner_command == "resume":
        return [*prefix, "--run-dir", output_directory, *controls]
    raise OperationsError("operational_argv_mismatch")


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


def _atomic_bytes_create(path: Path, payload: bytes) -> None:
    if path.exists():
        raise OperationsError("operational_evidence_collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
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
    resume_of_launch_record_path: Path | None = None,
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
        log_file_path = Path(log_path)
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
                    log_path,
                )
            )
            or any(
                ".." in path.parts
                for path in (
                    working_path,
                    python_path,
                    output_path,
                    authorized_output_path,
                    log_file_path,
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
            "log": log_file_path,
        }
        resolved_paths = _resolved_operational_paths(operational_paths)
        if resolved_paths["output"] != resolved_paths["authorized_output"]:
            raise OperationsError("operational_identity_mismatch")
        try:
            resolved_paths["output"].relative_to(resolved_paths["output_root"])
        except ValueError as exc:
            raise OperationsError("operational_working_directory_mismatch") from exc
        if (
            resolved_paths["python"] != resolved_paths["expected_python"]
            or output_path.name != authorization.get("run_id")
            or log_file_path.name != f"{authorization.get('run_id')}.log"
        ):
            raise OperationsError("operational_working_directory_mismatch")
        if (
            not unit.startswith("p7c4b2b-inner-")
            or unit.endswith(".service")
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
    elif execution_stage == "target-outer-projection-preflight":
        mode, task_set_digest, task_ids = _outer_control_identity(
            authorization=authorization,
            environment=environment,
            proposal=proposal,
            git_commit=git_commit,
        )
        authorized_output = authorization.get("output_directory")
        if (
            not isinstance(authorized_output, str)
            or not authorized_output
            or proposal.get("output_directory") != authorized_output
            or environment.get("output_directory") != authorized_output
        ):
            raise OperationsError("operational_identity_mismatch")
        working_path = Path(working_directory)
        if _working_tree_git_head(working_path) != git_commit:
            raise OperationsError("operational_git_provenance_mismatch")
        python_path = Path(python_executable)
        output_path = Path(output_directory)
        authorized_output_path = Path(authorized_output)
        expected_python = working_path / ".venv" / "bin" / "python"
        output_root = working_path / "artifacts"
        control_paths = {
            "environment": environment_path,
            "proposal": proposal_path,
            "authorization": authorization_path,
        }
        log_file_path = Path(log_path)
        if (
            not all(
                _is_absolute_operational_path(value)
                for value in (
                    working_directory,
                    python_executable,
                    output_directory,
                    authorized_output,
                    log_path,
                    *(str(path) for path in control_paths.values()),
                )
            )
            or any(
                ".." in path.parts
                for path in (
                    working_path,
                    python_path,
                    output_path,
                    authorized_output_path,
                    log_file_path,
                    *control_paths.values(),
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
            "output_root": output_root,
            "log": log_file_path,
            **control_paths,
        }
        resolved_paths = _resolved_operational_paths(operational_paths)
        expected_log_path = (
            resolved_paths["working"]
            / "artifacts"
            / "p7c4b2c-operations"
            / f"{unit}.log"
        )
        if (
            resolved_paths["python"] != resolved_paths["expected_python"]
            or resolved_paths["output"] != resolved_paths["authorized_output"]
            or resolved_paths["log"] != expected_log_path
            or len({resolved_paths[name] for name in ("output", "log", *control_paths)})
            != 5
        ):
            raise OperationsError("operational_identity_mismatch")
        try:
            resolved_paths["output"].relative_to(resolved_paths["output_root"])
        except ValueError as exc:
            raise OperationsError("operational_working_directory_mismatch") from exc
        runner_command = argv[3] if len(argv) >= 4 else ""
        expected_argv = _outer_expected_argv(
            runner_command=runner_command,
            python_executable=python_executable,
            mode=mode,
            output_directory=output_directory,
            environment_path=environment_path,
            proposal_path=proposal_path,
            authorization_path=authorization_path,
        )
        if argv != expected_argv:
            raise OperationsError("operational_argv_mismatch")
        if runner_command == "run" and output_path.exists():
            raise OperationsError("output_collision")
        if runner_command == "resume" and not output_path.is_dir():
            raise OperationsError("resume_precheck_missing_required_artifact")
        resume_of_launch_record = None
        if runner_command == "resume":
            if resume_of_launch_record_path is None:
                raise OperationsError("resume_launch_record_missing")
            original_launch, original_launch_hash = _json_input(
                resume_of_launch_record_path
            )
            if (
                original_launch.get("execution_stage") != execution_stage
                or original_launch.get("artifact_type") != artifact_type
                or original_launch.get("runner_command") != "run"
                or original_launch.get("mode") != mode
                or original_launch.get("source_git_commit") != git_commit
                or original_launch.get("resolved_output_directory")
                != str(resolved_paths["output"])
                or original_launch.get("record_digest")
                != _record_digest(original_launch, "record_digest")
                or original_launch.get("environment", {}).get("artifact_digest")
                != environment.get("environment_digest")
                or original_launch.get("proposal", {}).get("artifact_digest")
                != proposal.get("proposal_digest")
                or original_launch.get("authorization", {}).get("artifact_digest")
                != authorization.get("authorization_digest")
                or original_launch.get("systemd_unit") == unit
            ):
                raise OperationsError("resume_launch_identity_mismatch")
            resume_of_launch_record = {
                "path": str(resume_of_launch_record_path.resolve()),
                "sha256": original_launch_hash,
                "record_digest": original_launch["record_digest"],
                "systemd_unit": original_launch["systemd_unit"],
            }
        elif resume_of_launch_record_path is not None:
            raise OperationsError("operational_argv_mismatch")
        mode_token = "p1" if mode == "cpu_parallel_1" else "p2"
        if (
            not unit.startswith(f"p7c4b2c-outer-{mode_token}-")
            or output_path.name not in unit
            or not unit.endswith(".service")
        ):
            raise OperationsError("systemd_unit_mismatch")
        outer_environment_binding = _typed_input(environment_path, "environment_digest")
        outer_proposal_binding = _typed_input(proposal_path, "proposal_digest")
        outer_authorization_binding = _typed_input(
            authorization_path, "authorization_digest"
        )
        _revalidate_operational_paths(operational_paths, resolved_paths)
    value = {
        "schema_version": (
            OUTER_LAUNCH_RECORD_SCHEMA_VERSION
            if execution_stage == "target-outer-projection-preflight"
            else LAUNCH_RECORD_SCHEMA_VERSION
        ),
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
    elif execution_stage == "target-outer-projection-preflight":
        value.update(
            {
                "execution_stage": execution_stage,
                "protocol_stage": AUTHORIZATION_STAGE_BY_OPERATION_STAGE[
                    execution_stage
                ],
                "runner_command": runner_command,
                "mode": mode,
                "run_id": output_path.name,
                "resolved_working_directory": str(resolved_paths["working"]),
                "resolved_python_executable": str(resolved_paths["python"]),
                "resolved_output_directory": str(resolved_paths["output"]),
                "resolved_log_path": str(resolved_paths["log"]),
                "task_ids": task_ids,
                "task_set_digest": task_set_digest,
                "environment": outer_environment_binding,
                "proposal": outer_proposal_binding,
                "authorization": outer_authorization_binding,
            }
        )
        if resume_of_launch_record is not None:
            value["resume_of_launch_record"] = resume_of_launch_record
        value["record_digest"] = _record_digest(value, "record_digest")
    _atomic_create(record_path, value)
    return value


def _submission_claim_path(launch_record_path: Path) -> Path:
    return launch_record_path.with_name(
        f".{launch_record_path.name}.submission-claim.json"
    )


def _submission_attempt_path(launch_record_path: Path) -> Path:
    return launch_record_path.with_name(
        f".{launch_record_path.name}.submission-attempt.json"
    )


def _submission_capture_path(launch_record_path: Path) -> Path:
    return launch_record_path.with_name(
        f".{launch_record_path.name}.submission-capture.json"
    )


def _validate_outer_launch_for_submission(record: dict[str, Any]) -> None:
    """Revalidate a fresh outer launch independently at the submit boundary."""
    expected_fields = OUTER_LAUNCH_RECORD_FIELDS | (
        {"resume_of_launch_record"}
        if record.get("runner_command") == "resume"
        else set()
    )
    if (
        set(record) != expected_fields
        or record.get("schema_version") != OUTER_LAUNCH_RECORD_SCHEMA_VERSION
        or record.get("artifact_type")
        != EXECUTION_ARTIFACT_TYPES["target-outer-projection-preflight"][0]
        or record.get("execution_stage") != "target-outer-projection-preflight"
        or record.get("protocol_stage") != "target_projection_preflight"
        or record.get("execution_class") != "target_preflight"
        or record.get("submission_state") != "prepared_not_submitted"
        or not isinstance(record.get("source_git_commit"), str)
        or GIT_COMMIT_RE.fullmatch(record["source_git_commit"]) is None
        or record.get("record_digest") != _record_digest(record, "record_digest")
    ):
        raise OperationsError("launch_record_state_invalid")
    try:
        environment_binding = record["environment"]
        proposal_binding = record["proposal"]
        authorization_binding = record["authorization"]
        if not all(
            isinstance(binding, dict)
            for binding in (
                environment_binding,
                proposal_binding,
                authorization_binding,
            )
        ):
            raise OperationsError("launch_record_state_invalid")
        environment_path = Path(environment_binding["path"])
        proposal_path = Path(proposal_binding["path"])
        authorization_path = Path(authorization_binding["path"])
        environment, _ = _json_input(environment_path)
        proposal, _ = _json_input(proposal_path)
        authorization, _ = _json_input(authorization_path)
        mode, task_set_digest, task_ids = _outer_control_identity(
            authorization=authorization,
            environment=environment,
            proposal=proposal,
            git_commit=record["source_git_commit"],
        )
        if (
            environment_binding != _typed_input(environment_path, "environment_digest")
            or proposal_binding != _typed_input(proposal_path, "proposal_digest")
            or authorization_binding
            != _typed_input(authorization_path, "authorization_digest")
            or record.get("mode") != mode
            or record.get("task_ids") != task_ids
            or record.get("task_set_digest") != task_set_digest
        ):
            raise OperationsError("operational_identity_mismatch")
        working_path = Path(record["working_directory"])
        python_path = Path(record["python_executable"])
        output_path = Path(record["output_directory"])
        log_path = Path(record["log_path"])
        paths = {
            "working": working_path,
            "python": python_path,
            "expected_python": working_path / ".venv" / "bin" / "python",
            "output": output_path,
            "authorized_output": Path(authorization["output_directory"]),
            "output_root": working_path / "artifacts",
            "log": log_path,
            "environment": environment_path,
            "proposal": proposal_path,
            "authorization": authorization_path,
        }
        if (
            not all(_is_absolute_operational_path(str(path)) for path in paths.values())
            or any(".." in path.parts for path in paths.values())
            or output_path.is_symlink()
        ):
            raise OperationsError("operational_working_directory_mismatch")
        resolved = _resolved_operational_paths(paths)
        expected_log_path = (
            resolved["working"]
            / "artifacts"
            / "p7c4b2c-operations"
            / f"{record.get('systemd_unit')}.log"
        )
        if (
            resolved["python"] != resolved["expected_python"]
            or resolved["output"] != resolved["authorized_output"]
            or record.get("resolved_working_directory") != str(resolved["working"])
            or record.get("resolved_python_executable") != str(resolved["python"])
            or record.get("resolved_output_directory") != str(resolved["output"])
            or record.get("resolved_log_path") != str(resolved["log"])
            or resolved["log"] != expected_log_path
            or len(
                {
                    resolved[name]
                    for name in (
                        "output",
                        "log",
                        "environment",
                        "proposal",
                        "authorization",
                    )
                }
            )
            != 5
            or record.get("run_id") != output_path.name
        ):
            raise OperationsError("operational_identity_mismatch")
        resolved["output"].relative_to(resolved["output_root"])
        expected_argv = _outer_expected_argv(
            runner_command=record["runner_command"],
            python_executable=record["python_executable"],
            mode=mode,
            output_directory=record["output_directory"],
            environment_path=environment_path,
            proposal_path=proposal_path,
            authorization_path=authorization_path,
        )
        if record.get("argv") != expected_argv:
            raise OperationsError("operational_argv_mismatch")
        mode_token = "p1" if mode == "cpu_parallel_1" else "p2"
        unit = record.get("systemd_unit")
        if (
            not isinstance(unit, str)
            or not unit.startswith(f"p7c4b2c-outer-{mode_token}-")
            or output_path.name not in unit
            or not unit.endswith(".service")
        ):
            raise OperationsError("systemd_unit_mismatch")
        if record["runner_command"] == "run":
            if output_path.exists() or "resume_of_launch_record" in record:
                raise OperationsError("output_collision")
        elif record["runner_command"] == "resume":
            original = record["resume_of_launch_record"]
            if not isinstance(original, dict) or set(original) != {
                "path",
                "sha256",
                "record_digest",
                "systemd_unit",
            }:
                raise OperationsError("resume_launch_identity_mismatch")
            original_path = Path(original["path"])
            original_record, original_hash = _json_input(original_path)
            if (
                not output_path.is_dir()
                or original_hash != original["sha256"]
                or original_record.get("record_digest") != original["record_digest"]
                or original_record.get("systemd_unit") != original["systemd_unit"]
                or original_record.get("runner_command") != "run"
                or original_record.get("source_git_commit")
                != record["source_git_commit"]
                or original_record.get("resolved_output_directory")
                != str(resolved["output"])
                or original_record.get("record_digest")
                != _record_digest(original_record, "record_digest")
                or original_record.get("systemd_unit") == unit
            ):
                raise OperationsError("resume_launch_identity_mismatch")
        else:
            raise OperationsError("operational_argv_mismatch")
        if _working_tree_git_head(resolved["working"]) != record["source_git_commit"]:
            raise OperationsError("operational_git_provenance_mismatch")
        _revalidate_operational_paths(paths, resolved)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, OperationsError):
            raise
        raise OperationsError("launch_record_state_invalid") from exc


def _validate_b2b_launch_for_submission(record: dict[str, Any]) -> None:
    """Revalidate the typed B2b launch at the same submit boundary as outer."""
    if (
        set(record) != B2B_LAUNCH_RECORD_FIELDS
        or record.get("schema_version") != LAUNCH_RECORD_SCHEMA_VERSION
        or record.get("artifact_type")
        != EXECUTION_ARTIFACT_TYPES["target-inner-preflight"][0]
        or record.get("execution_stage") != "target-inner-preflight"
        or record.get("execution_class") != "target_preflight"
        or record.get("submission_state") != "prepared_not_submitted"
        or record.get("runner_command") not in {"run", "resume"}
        or not isinstance(record.get("source_git_commit"), str)
        or GIT_COMMIT_RE.fullmatch(record["source_git_commit"]) is None
        or record.get("record_digest") != _record_digest(record, "record_digest")
        or not isinstance(record.get("systemd_unit"), str)
        or not record["systemd_unit"].startswith("p7c4b2b-inner-")
        or record["systemd_unit"].endswith(".service")
        or record.get("run_id") not in record["systemd_unit"]
    ):
        raise OperationsError("launch_record_state_invalid")
    try:
        profile_binding = record["machine_profile"]
        environment_binding = record["environment"]
        proposal_binding = record["proposal"]
        authorization_binding = record["authorization"]
        if not all(
            isinstance(binding, dict)
            for binding in (
                profile_binding,
                environment_binding,
                proposal_binding,
                authorization_binding,
            )
        ):
            raise OperationsError("launch_record_state_invalid")
        profile_path = Path(profile_binding["path"])
        environment_path = Path(environment_binding["path"])
        proposal_path = Path(proposal_binding["path"])
        authorization_path = Path(authorization_binding["path"])
        profile, _ = _json_input(profile_path)
        environment, _ = _json_input(environment_path)
        proposal, _ = _json_input(proposal_path)
        authorization, _ = _json_input(authorization_path)
        if (
            profile_binding != _typed_input(profile_path, "profile_digest")
            or environment_binding != _typed_input(environment_path, "environment_digest")
            or proposal_binding != _typed_input(proposal_path, "proposal_digest")
            or authorization_binding
            != _typed_input(authorization_path, "authorization_digest")
            or authorization.get("execution_stage")
            != AUTHORIZATION_STAGE_BY_OPERATION_STAGE["target-inner-preflight"]
            or environment.get("execution_stage")
            != AUTHORIZATION_STAGE_BY_OPERATION_STAGE["target-inner-preflight"]
            or proposal.get("execution_stage")
            != AUTHORIZATION_STAGE_BY_OPERATION_STAGE["target-inner-preflight"]
            or authorization.get("source_git_commit") != record["source_git_commit"]
            or environment.get("source_git_commit") != record["source_git_commit"]
            or profile.get("git_commit") != record["source_git_commit"]
            or profile.get("profile_digest")
            != authorization.get("machine_profile_digest")
            or not isinstance(authorization.get("normalized_output_directory"), str)
            or not authorization.get("normalized_output_directory")
        ):
            raise OperationsError("operational_identity_mismatch")
        working_path = Path(record["working_directory"])
        python_path = Path(record["python_executable"])
        output_path = Path(record["output_directory"])
        authorized_output_path = Path(authorization["normalized_output_directory"])
        log_path = Path(record["log_path"])
        paths = {
            "working": working_path,
            "python": python_path,
            "expected_python": working_path / ".venv" / "bin" / "python",
            "output": output_path,
            "authorized_output": authorized_output_path,
            "output_root": working_path / "artifacts" / "p7c4b2b-compute-preflight",
            "log": log_path,
        }
        raw_path_values = (
            record["working_directory"],
            record["python_executable"],
            record["output_directory"],
            authorization["normalized_output_directory"],
            record["log_path"],
        )
        if (
            not all(_is_absolute_operational_path(value) for value in raw_path_values)
            or any(".." in PurePosixPath(value).parts for value in raw_path_values)
            or output_path.is_symlink()
        ):
            raise OperationsError("operational_working_directory_mismatch")
        resolved = _resolved_operational_paths(paths)
        if (
            resolved["python"] != resolved["expected_python"]
            or resolved["output"] != resolved["authorized_output"]
            or record.get("run_id") != output_path.name
            or log_path.name != f"{record.get('run_id')}.log"
        ):
            raise OperationsError("operational_identity_mismatch")
        resolved["output"].relative_to(resolved["output_root"])
        expected_argv = [
            record["python_executable"],
            "-m",
            "creditrep.experiments.p7c4b2b_cli",
            record["runner_command"],
            "--mode",
            authorization.get("mode"),
            "--profile",
            str(profile_path),
            "--target-machine-asserted",
            "--target-environment",
            str(environment_path),
            "--authorization-proposal",
            str(proposal_path),
            "--effective-authorization",
            str(authorization_path),
            "--output-dir",
            record["output_directory"],
        ]
        if record.get("argv") != expected_argv:
            raise OperationsError("operational_argv_mismatch")
        if _working_tree_git_head(resolved["working"]) != record["source_git_commit"]:
            raise OperationsError("operational_git_provenance_mismatch")
        _revalidate_operational_paths(paths, resolved)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, OperationsError):
            raise
        raise OperationsError("launch_record_state_invalid") from exc


def _hardened_submission_stage(record: dict[str, Any]) -> str:
    stage = record.get("execution_stage")
    if stage == "target-outer-projection-preflight":
        _validate_outer_launch_for_submission(record)
    elif stage == "target-inner-preflight":
        _validate_b2b_launch_for_submission(record)
    else:
        raise OperationsError("launch_record_state_invalid")
    return stage


def _submission_artifact_type(stage: str, role: str) -> str:
    if role not in {"attempt", "capture", "result"}:
        raise OperationsError("submission_artifact_role_invalid")
    try:
        prefix = SUBMISSION_STAGE_PREFIXES[stage]
    except KeyError as exc:
        raise OperationsError("launch_record_state_invalid") from exc
    return f"{prefix}_submission_{role}"


def _submission_stage_policy(stage: str) -> dict[str, Any]:
    try:
        return SUBMISSION_STAGE_POLICIES[stage]
    except KeyError as exc:
        raise OperationsError("launch_record_state_invalid") from exc


def _submission_claim_is_valid(
    *, record: dict[str, Any], launch_record_path: Path, launch_hash: str, claim: dict[str, Any]
) -> bool:
    stage = record["execution_stage"]
    policy = _submission_stage_policy(stage)
    fields = policy["claim_fields"]
    schema = policy["claim_schema"]
    return (
        set(claim) == fields
        and claim.get("schema_version") == schema
        and claim.get("artifact_type") == SUBMISSION_CLAIM_ARTIFACT_TYPES[stage]
        and claim.get("execution_stage") == stage
        and claim.get("launch_record_path") == str(launch_record_path.resolve())
        and claim.get("launch_record_sha256") == launch_hash
        and (stage != "target-outer-projection-preflight" or claim.get("launch_record_digest") == record.get("record_digest"))
        and claim.get("systemd_unit") == record.get("systemd_unit")
        and claim.get("submission_state") == "claimed_not_submitted"
        and claim.get("claim_digest") == _record_digest(claim, "claim_digest")
    )


def _create_submission_attempt(launch_record_path: Path) -> dict[str, Any]:
    """Atomically record the one permitted systemd invocation."""
    record, launch_hash = _json_input(launch_record_path)
    stage = _hardened_submission_stage(record)
    claim_path = _submission_claim_path(launch_record_path)
    claim, claim_hash = _json_input(claim_path)
    if not _submission_claim_is_valid(
        record=record, launch_record_path=launch_record_path, launch_hash=launch_hash, claim=claim
    ):
        raise OperationsError("submission_claim_invalid")
    value = {
        "schema_version": 1,
        "artifact_type": _submission_artifact_type(stage, "attempt"),
        "attempted_at": _utc_now(),
        "launch_record_path": str(launch_record_path.resolve()),
        "launch_record_sha256": launch_hash,
        "launch_record_digest": record.get("record_digest"),
        "submission_claim_path": str(claim_path.resolve()),
        "submission_claim_sha256": claim_hash,
        "submission_claim_digest": claim.get("claim_digest"),
        "systemd_unit": record.get("systemd_unit"),
        "attempt_state": "submission_invocation_committed",
    }
    value["attempt_digest"] = _record_digest(value, "attempt_digest")
    try:
        _atomic_create(_submission_attempt_path(launch_record_path), value)
    except OperationsError as exc:
        if str(exc) == "operational_evidence_collision":
            raise OperationsError("submission_already_attempted") from exc
        raise
    return value


def _create_submission_capture(
    *,
    launch_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    exit_code_path: Path,
    systemd_run_exit_code: int,
) -> dict[str, Any]:
    """Atomically bind the raw bytes captured by the committed invocation."""
    saved_exit_code, exit_code_hash = _read_exit_code_evidence(exit_code_path)
    if saved_exit_code != systemd_run_exit_code:
        raise OperationsError("systemd_run_exit_code_evidence_mismatch")
    record, launch_hash = _json_input(launch_record_path)
    stage = _hardened_submission_stage(record)
    claim_path = _submission_claim_path(launch_record_path)
    claim, claim_hash = _json_input(claim_path)
    if not _submission_claim_is_valid(
        record=record, launch_record_path=launch_record_path, launch_hash=launch_hash, claim=claim
    ):
        raise OperationsError("submission_claim_invalid")
    attempt_path = _submission_attempt_path(launch_record_path)
    attempt, attempt_hash = _json_input(attempt_path)
    if (
        set(attempt) != _submission_stage_policy(stage)["attempt_fields"]
        or attempt.get("schema_version") != 1
        or attempt.get("artifact_type") != _submission_artifact_type(stage, "attempt")
        or attempt.get("launch_record_path") != str(launch_record_path.resolve())
        or attempt.get("launch_record_sha256") != launch_hash
        or attempt.get("launch_record_digest") != record.get("record_digest")
        or attempt.get("submission_claim_path") != str(claim_path.resolve())
        or attempt.get("submission_claim_sha256") != claim_hash
        or attempt.get("submission_claim_digest") != claim.get("claim_digest")
        or attempt.get("systemd_unit") != record.get("systemd_unit")
        or attempt.get("attempt_state") != "submission_invocation_committed"
        or attempt.get("attempt_digest") != _record_digest(attempt, "attempt_digest")
    ):
        raise OperationsError("submission_attempt_invalid")
    value = {
        "schema_version": 1,
        "artifact_type": _submission_artifact_type(stage, "capture"),
        "created_at": _utc_now(),
        "execution_stage": stage,
        "launch_record_path": str(launch_record_path.resolve()),
        "launch_record_sha256": launch_hash,
        "launch_record_digest": record["record_digest"],
        "submission_claim_path": str(claim_path.resolve()),
        "submission_claim_sha256": claim_hash,
        "submission_claim_digest": claim["claim_digest"],
        "submission_attempt_path": str(attempt_path.resolve()),
        "submission_attempt_sha256": attempt_hash,
        "submission_attempt_digest": attempt["attempt_digest"],
        "systemd_unit": record["systemd_unit"],
        "stdout_path": str(stdout_path.resolve()),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_path": str(stderr_path.resolve()),
        "stderr_sha256": _sha256_file(stderr_path),
        "exit_code_path": str(exit_code_path.resolve()),
        "exit_code_sha256": exit_code_hash,
        "systemd_run_exit_code": systemd_run_exit_code,
    }
    value["capture_digest"] = _record_digest(value, "capture_digest")
    _atomic_create(_submission_capture_path(launch_record_path), value)
    return value


def create_submission_claim(
    *, launch_record_path: Path, receipt_path: Path
) -> dict[str, Any]:
    record, launch_hash = _json_input(launch_record_path)
    stage = record.get("execution_stage")
    valid_launch_schemas = (
        {LAUNCH_RECORD_SCHEMA_VERSION, OUTER_LAUNCH_RECORD_SCHEMA_VERSION}
        if stage == "target-outer-projection-preflight"
        else {LAUNCH_RECORD_SCHEMA_VERSION}
    )
    if (
        stage not in SUBMISSION_CLAIM_ARTIFACT_TYPES
        or record.get("artifact_type") != EXECUTION_ARTIFACT_TYPES[stage][0]
        or record.get("schema_version") not in valid_launch_schemas
        or record.get("submission_state") != "prepared_not_submitted"
        or record.get("record_digest") != _record_digest(record, "record_digest")
    ):
        raise OperationsError("launch_record_state_invalid")
    if stage == "target-inner-preflight":
        _validate_b2b_launch_for_submission(record)
    elif stage == "target-outer-projection-preflight":
        expected_fields = OUTER_LAUNCH_RECORD_FIELDS | (
            {"resume_of_launch_record"}
            if record.get("runner_command") == "resume"
            else set()
        )
        if set(record) != expected_fields:
            raise OperationsError("launch_record_state_invalid")
        if record.get("schema_version") == OUTER_LAUNCH_RECORD_SCHEMA_VERSION:
            _validate_outer_launch_for_submission(record)
    value = {
        "schema_version": (
            OUTER_SUBMISSION_CLAIM_SCHEMA_VERSION
            if stage == "target-outer-projection-preflight"
            and record.get("schema_version") == OUTER_LAUNCH_RECORD_SCHEMA_VERSION
            else SUBMISSION_CLAIM_SCHEMA_VERSION
        ),
        "artifact_type": SUBMISSION_CLAIM_ARTIFACT_TYPES[stage],
        "execution_stage": stage,
        "claimed_at": _utc_now(),
        "launch_record_path": str(launch_record_path.resolve()),
        "launch_record_sha256": launch_hash,
        "receipt_path": str(receipt_path.resolve()),
        "systemd_unit": record["systemd_unit"],
        "submission_state": "claimed_not_submitted",
    }
    if stage == "target-outer-projection-preflight":
        value["launch_record_digest"] = record["record_digest"]
    value["claim_digest"] = _record_digest(value, "claim_digest")
    try:
        _atomic_create(_submission_claim_path(launch_record_path), value)
    except OperationsError as exc:
        if str(exc) == "operational_evidence_collision":
            raise OperationsError("duplicate_submission") from exc
        raise
    return value


def _expected_systemd_reported_unit(execution_stage: str, stored_unit: str) -> str:
    """Map the persisted project identity to systemd's reported identity."""
    if execution_stage == "target-inner-preflight":
        if stored_unit.endswith(".service"):
            raise OperationsError("systemd_unit_mismatch")
        return f"{stored_unit}.service"
    if execution_stage == "target-outer-projection-preflight":
        return stored_unit
    raise OperationsError("launch_record_state_invalid")


def _parse_systemd_run_output(
    stdout: str, stderr: str, expected_unit: str, execution_stage: str
) -> str:
    """Resolve one unambiguous systemd submission identity from both channels.

    One valid observation in either channel is sufficient.  The same valid
    identity may be echoed once in each channel; different identities, or more
    than one identity in one channel, are rejected rather than guessed.
    """
    pattern = re.compile(
        r"Running as unit:\s*([^;\r\n]+);\s*invocation ID:\s*([^\s\r\n]+)"
    )
    observations: list[tuple[str, str]] = []
    for channel in (stdout, stderr):
        matches = pattern.findall(channel)
        # Every identity-looking marker must be consumed by one complete
        # observation.  This rejects truncated or orphaned competing evidence
        # without treating unrelated stderr diagnostics as identity evidence.
        if (
            channel.count("Running as unit:") != len(matches)
            or channel.count("invocation ID:") != len(matches)
            or len(matches) > 1
        ):
            raise OperationsError("systemd_run_invocation_id_count_invalid")
        if matches:
            unit, invocation_id = (item.strip() for item in matches[0])
            if INVOCATION_ID_RE.fullmatch(invocation_id) is None:
                raise OperationsError("invocation_id_malformed")
            observations.append((unit, invocation_id))
    if not observations:
        raise OperationsError("systemd_run_invocation_id_count_invalid")
    if len(set(observations)) != 1:
        raise OperationsError("systemd_run_output_conflict")
    unit, invocation_id = observations[0]
    if unit != _expected_systemd_reported_unit(execution_stage, expected_unit):
        raise OperationsError("systemd_unit_mismatch")
    return invocation_id


def _assert_original_unit_inactive(record: dict[str, Any]) -> None:
    """Recheck the original unit immediately before a resume submission."""
    if record.get("runner_command") != "resume":
        return
    original_unit = record["resume_of_launch_record"]["systemd_unit"]
    command = [
        "systemctl",
        "--user",
        "show",
        original_unit,
        "--property=LoadState,ActiveState,MainPID",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=15, check=False
        )
        output = completed.stdout.decode("utf-8")
    except (OSError, UnicodeDecodeError, subprocess.TimeoutExpired) as exc:
        raise OperationsError("resume_precheck_unit_state_unavailable") from exc
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values:
            raise OperationsError("resume_precheck_unit_state_invalid")
        values[key] = value
    if set(values) != {"LoadState", "ActiveState", "MainPID"}:
        raise OperationsError("resume_precheck_unit_state_invalid")
    if values["LoadState"] == "not-found":
        if values["ActiveState"] not in {"", "inactive"} or values["MainPID"] not in {
            "",
            "0",
        }:
            raise OperationsError("resume_precheck_unit_active")
        return
    if values["ActiveState"] not in {"inactive", "failed"} or values["MainPID"] not in {
        "",
        "0",
    }:
        raise OperationsError("resume_precheck_unit_active")


def submit_systemd_run(
    *,
    result_path: Path,
    launch_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    exit_code_path: Path,
) -> dict[str, Any]:
    """Submit one claimed hardened launch and durably capture its process result."""
    record, launch_hash = _json_input(launch_record_path)
    _hardened_submission_stage(record)
    claim_path = _submission_claim_path(launch_record_path)
    claim, _claim_hash = _json_input(claim_path)
    if not _submission_claim_is_valid(
        record=record, launch_record_path=launch_record_path, launch_hash=launch_hash, claim=claim
    ):
        raise OperationsError("submission_claim_invalid")
    if _submission_attempt_path(launch_record_path).exists():
        raise OperationsError("submission_already_attempted")
    evidence_paths = (
        result_path, stdout_path, stderr_path, exit_code_path,
        _submission_capture_path(launch_record_path),
    )
    if any(path.exists() for path in evidence_paths):
        raise OperationsError("operational_evidence_collision")
    command = [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={record['systemd_unit']}",
        f"--working-directory={record['working_directory']}",
        f"--property=StandardOutput=append:{record['log_path']}",
        f"--property=StandardError=append:{record['log_path']}",
        *record["argv"],
    ]
    _assert_original_unit_inactive(record)
    _create_submission_attempt(launch_record_path)
    previous_sighup = None
    if hasattr(signal, "SIGHUP"):
        previous_sighup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=60, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OperationsError("systemd_run_capture_failed") from exc
        _atomic_bytes_create(stdout_path, completed.stdout)
        _atomic_bytes_create(stderr_path, completed.stderr)
        _atomic_bytes_create(
            exit_code_path, f"{completed.returncode}\n".encode("ascii")
        )
        _create_submission_capture(
            launch_record_path=launch_record_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            exit_code_path=exit_code_path,
            systemd_run_exit_code=completed.returncode,
        )
        return create_submission_result(
            result_path=result_path,
            launch_record_path=launch_record_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            exit_code_path=exit_code_path,
            systemd_run_exit_code=completed.returncode,
        )
    finally:
        if previous_sighup is not None:
            signal.signal(signal.SIGHUP, previous_sighup)


def create_submission_result(
    *,
    result_path: Path,
    launch_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    exit_code_path: Path,
    systemd_run_exit_code: int,
) -> dict[str, Any]:
    """Bind an outer systemd-run outcome to immutable saved process evidence."""
    if (
        not isinstance(systemd_run_exit_code, int)
        or isinstance(systemd_run_exit_code, bool)
        or not 0 <= systemd_run_exit_code <= 255
    ):
        raise OperationsError("invalid_systemd_run_exit_code")
    saved_exit_code, exit_code_hash = _read_exit_code_evidence(exit_code_path)
    if saved_exit_code != systemd_run_exit_code:
        raise OperationsError("systemd_run_exit_code_evidence_mismatch")
    record, launch_hash = _json_input(launch_record_path)
    stage = _hardened_submission_stage(record)
    claim_path = _submission_claim_path(launch_record_path)
    claim, claim_hash = _json_input(claim_path)
    if not _submission_claim_is_valid(
        record=record, launch_record_path=launch_record_path, launch_hash=launch_hash, claim=claim
    ):
        raise OperationsError("submission_claim_invalid")
    attempt_path = _submission_attempt_path(launch_record_path)
    attempt, attempt_hash = _json_input(attempt_path)
    if (
        set(attempt) != _submission_stage_policy(stage)["attempt_fields"]
        or attempt.get("schema_version") != 1
        or attempt.get("artifact_type") != _submission_artifact_type(stage, "attempt")
        or attempt.get("launch_record_path") != str(launch_record_path.resolve())
        or attempt.get("launch_record_sha256") != launch_hash
        or attempt.get("launch_record_digest") != record.get("record_digest")
        or attempt.get("submission_claim_path") != str(claim_path.resolve())
        or attempt.get("submission_claim_sha256") != claim_hash
        or attempt.get("submission_claim_digest") != claim.get("claim_digest")
        or attempt.get("systemd_unit") != record.get("systemd_unit")
        or attempt.get("attempt_state") != "submission_invocation_committed"
        or attempt.get("attempt_digest") != _record_digest(attempt, "attempt_digest")
    ):
        raise OperationsError("submission_attempt_invalid")
    capture_path = _submission_capture_path(launch_record_path)
    capture, capture_hash = _json_input(capture_path)
    if (
        set(capture) != _submission_stage_policy(stage)["capture_fields"]
        or capture.get("schema_version") != 1
        or capture.get("artifact_type") != _submission_artifact_type(stage, "capture")
        or capture.get("capture_digest") != _record_digest(capture, "capture_digest")
        or capture.get("execution_stage") != stage
        or capture.get("launch_record_path") != str(launch_record_path.resolve())
        or capture.get("launch_record_sha256") != launch_hash
        or capture.get("launch_record_digest") != record.get("record_digest")
        or capture.get("submission_claim_path") != str(claim_path.resolve())
        or capture.get("submission_claim_sha256") != claim_hash
        or capture.get("submission_claim_digest") != claim.get("claim_digest")
        or capture.get("submission_attempt_path") != str(attempt_path.resolve())
        or capture.get("submission_attempt_sha256") != attempt_hash
        or capture.get("submission_attempt_digest") != attempt.get("attempt_digest")
        or capture.get("systemd_unit") != record.get("systemd_unit")
        or capture.get("stdout_path") != str(stdout_path.resolve())
        or capture.get("stdout_sha256") != _sha256_file(stdout_path)
        or capture.get("stderr_path") != str(stderr_path.resolve())
        or capture.get("stderr_sha256") != _sha256_file(stderr_path)
        or capture.get("exit_code_path") != str(exit_code_path.resolve())
        or capture.get("exit_code_sha256") != exit_code_hash
        or capture.get("systemd_run_exit_code") != systemd_run_exit_code
    ):
        raise OperationsError("submission_capture_invalid")
    try:
        stdout = stdout_path.read_text(encoding="utf-8")
        stderr = stderr_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        raise OperationsError("submission_output_invalid") from exc
    invocation_id = None
    if systemd_run_exit_code == 0:
        invocation_id = _parse_systemd_run_output(
            stdout, stderr, record["systemd_unit"], stage
        )
    elif any(marker in channel for channel in (stdout, stderr) for marker in (
        "Running as unit:", "invocation ID:"
    )):
        raise OperationsError("systemd_run_exit_code_mismatch")
    value = {
        "schema_version": _submission_stage_policy(stage)["result_schema"],
        "artifact_type": _submission_artifact_type(stage, "result"),
        "created_at": _utc_now(),
        "launch_record_path": str(launch_record_path.resolve()),
        "launch_record_sha256": launch_hash,
        "launch_record_digest": record["record_digest"],
        "submission_claim_path": str(claim_path.resolve()),
        "submission_claim_sha256": claim_hash,
        "submission_claim_digest": claim["claim_digest"],
        "submission_attempt_path": str(attempt_path.resolve()),
        "submission_attempt_sha256": attempt_hash,
        "submission_attempt_digest": attempt["attempt_digest"],
        "systemd_unit": record["systemd_unit"],
        "submission_capture_path": str(capture_path.resolve()),
        "submission_capture_sha256": capture_hash,
        "submission_capture_digest": capture["capture_digest"],
        "stdout_path": str(stdout_path.resolve()),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_path": str(stderr_path.resolve()),
        "stderr_sha256": _sha256_file(stderr_path),
        "systemd_run_exit_code": systemd_run_exit_code,
        "exit_code_path": str(exit_code_path.resolve()),
        "exit_code_sha256": exit_code_hash,
        "invocation_id": invocation_id,
        "submission_state": (
            "submitted" if systemd_run_exit_code == 0 else "submission_failed"
        ),
    }
    value["result_digest"] = _record_digest(value, "result_digest")
    _atomic_create(result_path, value)
    return value


def create_submission_receipt(
    *,
    receipt_path: Path,
    launch_record_path: Path,
    unit_snapshot: dict[str, Any] | None,
    systemd_run_exit_code: int,
    observed_unit: str | None = None,
    submission_result_path: Path | None = None,
    snapshot_attempt_path: Path | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(systemd_run_exit_code, int)
        or isinstance(systemd_run_exit_code, bool)
        or not 0 <= systemd_run_exit_code <= 255
    ):
        raise OperationsError("invalid_systemd_run_exit_code")
    record, launch_hash = _json_input(launch_record_path)
    stage = record.get("execution_stage", "target-canary")
    fresh_outer = (
        stage == "target-outer-projection-preflight"
        and record.get("schema_version") == OUTER_LAUNCH_RECORD_SCHEMA_VERSION
    )
    fresh_b2b = (
        stage == "target-inner-preflight"
        and record.get("schema_version") == LAUNCH_RECORD_SCHEMA_VERSION
        and record.get("record_digest") == _record_digest(record, "record_digest")
    )
    expected_launch_schema = (
        OUTER_LAUNCH_RECORD_SCHEMA_VERSION
        if fresh_outer
        else LAUNCH_RECORD_SCHEMA_VERSION
    )
    if stage not in EXECUTION_ARTIFACT_TYPES or (
        record.get("artifact_type") != EXECUTION_ARTIFACT_TYPES[stage][0]
        or record.get("schema_version") != expected_launch_schema
        or not isinstance(record.get("systemd_unit"), str)
        or record.get("submission_state") != "prepared_not_submitted"
    ):
        raise OperationsError("launch_record_state_invalid")
    if fresh_b2b and submission_result_path is None:
        raise OperationsError("submission_result_missing")
    if fresh_outer or fresh_b2b:
        hardened_stage = _hardened_submission_stage(record)
        if submission_result_path is None:
            raise OperationsError("submission_result_missing")
        result, result_hash = _json_input(submission_result_path)
        saved_exit_code, current_exit_code_hash = _read_exit_code_evidence(
            Path(result.get("exit_code_path", ""))
        )
        attempt_path = _submission_attempt_path(launch_record_path)
        attempt, attempt_hash = _json_input(attempt_path)
        capture_path = _submission_capture_path(launch_record_path)
        capture, capture_hash = _json_input(capture_path)
        if (
            set(result) != _submission_stage_policy(hardened_stage)["result_fields"]
            or result.get("schema_version") != _submission_stage_policy(hardened_stage)["result_schema"]
            or result.get("artifact_type") != _submission_artifact_type(hardened_stage, "result")
            or result.get("result_digest") != _record_digest(result, "result_digest")
            or result.get("launch_record_path") != str(launch_record_path.resolve())
            or result.get("launch_record_sha256") != launch_hash
            or result.get("launch_record_digest") != record.get("record_digest")
            or result.get("systemd_unit") != record.get("systemd_unit")
            or result.get("systemd_run_exit_code") != systemd_run_exit_code
            or saved_exit_code != systemd_run_exit_code
            or result.get("submission_claim_path")
            != str(_submission_claim_path(launch_record_path).resolve())
            or result.get("submission_attempt_path") != str(attempt_path.resolve())
            or attempt_hash != result.get("submission_attempt_sha256")
            or result.get("submission_capture_path") != str(capture_path.resolve())
            or result.get("submission_capture_sha256") != capture_hash
            or result.get("submission_capture_digest") != capture.get("capture_digest")
            or set(attempt) != _submission_stage_policy(hardened_stage)["attempt_fields"]
            or attempt.get("schema_version") != 1
            or attempt.get("artifact_type") != _submission_artifact_type(hardened_stage, "attempt")
            or attempt.get("launch_record_path") != str(launch_record_path.resolve())
            or attempt.get("launch_record_sha256") != launch_hash
            or attempt.get("launch_record_digest") != record.get("record_digest")
            or attempt.get("submission_claim_path")
            != str(_submission_claim_path(launch_record_path).resolve())
            or attempt.get("submission_claim_sha256")
            != result.get("submission_claim_sha256")
            or attempt.get("submission_claim_digest")
            != result.get("submission_claim_digest")
            or attempt.get("systemd_unit") != record.get("systemd_unit")
            or attempt.get("attempt_state") != "submission_invocation_committed"
            or attempt.get("attempt_digest")
            != _record_digest(attempt, "attempt_digest")
            or result.get("submission_attempt_digest") != attempt.get("attempt_digest")
            or set(capture) != _submission_stage_policy(hardened_stage)["capture_fields"]
            or capture.get("schema_version") != 1
            or capture.get("artifact_type") != _submission_artifact_type(hardened_stage, "capture")
            or capture.get("capture_digest") != _record_digest(capture, "capture_digest")
            or capture.get("execution_stage") != hardened_stage
            or capture.get("launch_record_path") != str(launch_record_path.resolve())
            or capture.get("launch_record_sha256") != launch_hash
            or capture.get("launch_record_digest") != record.get("record_digest")
            or capture.get("submission_claim_path") != str(_submission_claim_path(launch_record_path).resolve())
            or capture.get("submission_claim_sha256") != result.get("submission_claim_sha256")
            or capture.get("submission_claim_digest") != result.get("submission_claim_digest")
            or capture.get("submission_attempt_path") != str(attempt_path.resolve())
            or capture.get("submission_attempt_sha256") != attempt_hash
            or capture.get("submission_attempt_digest") != attempt.get("attempt_digest")
            or capture.get("systemd_unit") != record.get("systemd_unit")
            or capture.get("stdout_path") != result.get("stdout_path")
            or capture.get("stdout_sha256") != result.get("stdout_sha256")
            or capture.get("stderr_path") != result.get("stderr_path")
            or capture.get("stderr_sha256") != result.get("stderr_sha256")
            or capture.get("exit_code_path") != result.get("exit_code_path")
            or capture.get("exit_code_sha256") != result.get("exit_code_sha256")
            or capture.get("systemd_run_exit_code") != systemd_run_exit_code
            or current_exit_code_hash != result.get("exit_code_sha256")
            or _sha256_file(Path(result.get("stdout_path", "")))
            != result.get("stdout_sha256")
            or _sha256_file(Path(result.get("stderr_path", "")))
            != result.get("stderr_sha256")
        ):
            raise OperationsError("submission_result_mismatch")
        invocation_id = result.get("invocation_id")
        if systemd_run_exit_code == 0:
            stdout = Path(result["stdout_path"]).read_text(encoding="utf-8")
            stderr = Path(result["stderr_path"]).read_text(encoding="utf-8")
            if invocation_id != _parse_systemd_run_output(
                stdout, stderr, record["systemd_unit"], stage
            ):
                raise OperationsError("submission_result_mismatch")
            if result.get("submission_state") != "submitted":
                raise OperationsError("submission_result_mismatch")
        elif (
            invocation_id is not None
            or result.get("submission_state") != "submission_failed"
        ):
            raise OperationsError("submission_result_mismatch")
        if observed_unit != record.get("systemd_unit"):
            raise OperationsError("systemd_unit_mismatch")
        # The durable submit result is the authoritative recovery evidence for
        # B2b.  A collected transient unit is expected to be unavailable, so a
        # post-submit systemctl lookup must not be required to make a receipt.
        # Outer preflight retains its existing snapshot-evidence contract.
        if unit_snapshot is None and snapshot_attempt_path is None and not fresh_b2b:
            raise OperationsError("unit_snapshot_evidence_missing")
        snapshot_status = (
            "recovered_from_immutable_submission_result"
            if fresh_b2b and unit_snapshot is None and snapshot_attempt_path is None
            else "unavailable_empty_attempt"
        )
        snapshot_binding = None
        if snapshot_attempt_path is not None:
            snapshot_binding = {
                "path": str(snapshot_attempt_path.resolve()),
                "sha256": _sha256_file(snapshot_attempt_path),
            }
            if snapshot_attempt_path.stat().st_size == 0:
                snapshot_status = "unavailable_empty_attempt"
        if unit_snapshot is not None:
            if set(unit_snapshot) != UNIT_SNAPSHOT_FIELDS or any(
                not isinstance(item, str) for item in unit_snapshot.values()
            ):
                raise OperationsError("unit_snapshot_schema_mismatch")
            if unit_snapshot["LoadState"] == "not-found":
                if unit_snapshot["InvocationID"]:
                    raise OperationsError("unit_snapshot_invocation_id_mismatch")
                snapshot_status = "unavailable_unit_collected"
            else:
                if (
                    systemd_run_exit_code == 0
                    and unit_snapshot["InvocationID"] != invocation_id
                ):
                    raise OperationsError("unit_snapshot_invocation_id_mismatch")
                if systemd_run_exit_code != 0 and (
                    unit_snapshot["InvocationID"]
                    or unit_snapshot["LoadState"] == "loaded"
                ):
                    raise OperationsError("systemd_run_exit_code_mismatch")
                snapshot_status = "live_verified"
        claim_path = _submission_claim_path(launch_record_path)
        claim, claim_hash = _json_input(claim_path)
        if claim.get("receipt_path") != str(receipt_path.resolve()):
            raise OperationsError("duplicate_submission")
        if (not _submission_claim_is_valid(
            record=record, launch_record_path=launch_record_path, launch_hash=launch_hash, claim=claim
        ) or claim.get("claim_digest") != result.get("submission_claim_digest")
            or claim_hash != result.get("submission_claim_sha256")):
            raise OperationsError("submission_claim_invalid")
        value = {
            "schema_version": _submission_stage_policy(hardened_stage)["receipt_schema"],
            "artifact_type": EXECUTION_ARTIFACT_TYPES[stage][1],
            "submitted_at": result["created_at"],
            "receipt_created_at": _utc_now(),
            "execution_stage": stage,
            "launch_record_path": str(launch_record_path.resolve()),
            "launch_record_sha256": launch_hash,
            "launch_record_digest": record["record_digest"],
            "submission_claim_path": str(claim_path.resolve()),
            "submission_claim_sha256": claim_hash,
            "submission_claim_digest": claim["claim_digest"],
            "submission_result_path": str(submission_result_path.resolve()),
            "submission_result_sha256": result_hash,
            "submission_result_digest": result["result_digest"],
            "systemd_unit": record["systemd_unit"],
            "systemd_run_exit_code": systemd_run_exit_code,
            "invocation_id": invocation_id,
            "unit_snapshot": unit_snapshot,
            "unit_snapshot_status": snapshot_status,
            "unit_snapshot_attempt": snapshot_binding,
            "submission_state": result["submission_state"],
            "evidence_scope": "submission_outcome_only_not_compute_success",
        }
        value["receipt_digest"] = _record_digest(value, "receipt_digest")
        _atomic_create(receipt_path, value)
        return value
    if unit_snapshot is None or set(unit_snapshot) != UNIT_SNAPSHOT_FIELDS:
        raise OperationsError("unit_snapshot_schema_mismatch")
    if any(not isinstance(item, str) for item in unit_snapshot.values()):
        raise OperationsError("unit_snapshot_schema_mismatch")
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
    if stage in SUBMISSION_CLAIM_ARTIFACT_TYPES:
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
            or claim.get("artifact_type") != SUBMISSION_CLAIM_ARTIFACT_TYPES[stage]
            or claim.get("execution_stage") != stage
            or claim.get("launch_record_path") != str(launch_record_path.resolve())
            or claim.get("launch_record_sha256") != launch_hash
            or claim.get("systemd_unit") != record["systemd_unit"]
            or claim.get("submission_state") != "claimed_not_submitted"
            or claim.get("claim_digest") != _record_digest(claim, "claim_digest")
        ):
            raise OperationsError("submission_claim_invalid")
        if (
            stage == "target-outer-projection-preflight"
            and set(claim) != OUTER_SUBMISSION_CLAIM_FIELDS
        ):
            raise OperationsError("submission_claim_invalid")
        if stage == "target-outer-projection-preflight" and claim.get(
            "launch_record_digest"
        ) != record.get("record_digest"):
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
        if stage == "target-outer-projection-preflight":
            value["launch_record_digest"] = record["record_digest"]
        value["receipt_digest"] = _record_digest(value, "receipt_digest")
    _atomic_create(receipt_path, value)
    return value


def resume_precheck(
    run_dir: Path,
    *,
    launch_record_path: Path | None = None,
    unit_snapshot: dict[str, Any] | None = None,
    environment_path: Path | None = None,
    proposal_path: Path | None = None,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    operational_codes: list[str] = []
    controls = (environment_path, proposal_path, authorization_path)
    if unit_snapshot is not None:
        if set(unit_snapshot) != UNIT_SNAPSHOT_FIELDS or any(
            not isinstance(value, str) for value in unit_snapshot.values()
        ):
            operational_codes.append("unit_snapshot_schema_mismatch")
        elif unit_snapshot["ActiveState"] not in {
            "inactive",
            "failed",
        } or unit_snapshot["MainPID"] not in {"", "0"}:
            operational_codes.append("resume_precheck_unit_active")
    if launch_record_path is not None:
        try:
            launch, _launch_hash = _json_input(launch_record_path)
            if (
                launch.get("execution_stage") != "target-outer-projection-preflight"
                or launch.get("runner_command") != "run"
                or launch.get("resolved_output_directory")
                != str(_resolve_operational_path(run_dir))
                or launch.get("record_digest")
                != _record_digest(launch, "record_digest")
            ):
                operational_codes.append("resume_precheck_control_identity_mismatch")
            if unit_snapshot is None or any(path is None for path in controls):
                operational_codes.append("resume_precheck_operational_input_missing")
        except OperationsError:
            operational_codes.append("resume_precheck_control_identity_mismatch")
    if any(path is not None for path in controls):
        if any(path is None for path in controls):
            operational_codes.append("resume_precheck_control_identity_mismatch")
        else:
            try:
                environment, _ = _json_input(environment_path)
                proposal, _ = _json_input(proposal_path)
                authorization, _ = _json_input(authorization_path)
                manifest, _ = _json_input(run_dir / "manifest.json")
                provenance = manifest.get("authorization_provenance", {})
                if (
                    provenance.get("target_environment_digest")
                    != environment.get("environment_digest")
                    or provenance.get("proposal_digest")
                    != proposal.get("proposal_digest")
                    or provenance.get("authorization_digest")
                    != authorization.get("authorization_digest")
                    or manifest.get("run_id") != run_dir.name
                ):
                    operational_codes.append(
                        "resume_precheck_control_identity_mismatch"
                    )
            except OperationsError:
                operational_codes.append("resume_precheck_control_identity_mismatch")
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
            "reason_codes": sorted(
                set(
                    [
                        *operational_codes,
                        "resume_precheck_missing_required_artifact",
                    ]
                )
            ),
            "completed": None,
            "expected": None,
            "resume_requires_existing_authorization_revalidation": True,
        }
    # Import only after the cheap structural fail-closed checks. Launch-record
    # creation and malformed-run rejection must remain usable before the target
    # ML runtime is imported or validated.
    from creditrep.experiments.p7c4b2c_preflight import validate_artifacts

    report = validate_artifacts(run_dir)
    unsafe_report_codes = [
        code
        for code in report.get("reason_codes", [])
        if any(
            marker in code
            for marker in (
                "corrupt",
                "invalid",
                "malformed",
                "mismatch",
                "rollback",
                "runtime_state",
                "failure",
            )
        )
    ]
    completed, expected = report.get("completed"), report.get("expected")
    incomplete = (
        isinstance(completed, int)
        and isinstance(expected, int)
        and 0 <= completed < expected
    )
    eligible = (
        incomplete
        and not (run_dir / "COMPLETED.json").exists()
        and not operational_codes
        and not unsafe_report_codes
    )
    return {
        "valid": eligible,
        "reason_codes": sorted(
            set(
                [
                    *report.get("reason_codes", []),
                    *operational_codes,
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
            "submit-systemd-run",
            "create-submission-result",
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
    parser.add_argument("--unit-snapshot-file", type=Path)
    parser.add_argument("--systemd-run-exit-code", type=int)
    parser.add_argument("--systemd-run-exit-code-file", type=Path)
    parser.add_argument("--submission-result", type=Path)
    parser.add_argument("--systemd-run-stdout", type=Path)
    parser.add_argument("--systemd-run-stderr", type=Path)
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
                resume_of_launch_record_path=args.launch_record,
            )
        elif args.command == "create-submission-claim":
            if args.launch_record is None or args.receipt is None:
                raise OperationsError("missing_required_operational_argument")
            value = create_submission_claim(
                launch_record_path=args.launch_record,
                receipt_path=args.receipt,
            )
        elif args.command == "submit-systemd-run":
            if None in (
                args.submission_result,
                args.launch_record,
                args.systemd_run_stdout,
                args.systemd_run_stderr,
                args.systemd_run_exit_code_file,
            ):
                raise OperationsError("missing_required_operational_argument")
            value = submit_systemd_run(
                result_path=args.submission_result,
                launch_record_path=args.launch_record,
                stdout_path=args.systemd_run_stdout,
                stderr_path=args.systemd_run_stderr,
                exit_code_path=args.systemd_run_exit_code_file,
            )
        elif args.command == "create-submission-result":
            if None in (
                args.submission_result,
                args.launch_record,
                args.systemd_run_stdout,
                args.systemd_run_stderr,
                args.systemd_run_exit_code_file,
                args.systemd_run_exit_code,
            ):
                raise OperationsError("missing_required_operational_argument")
            value = create_submission_result(
                result_path=args.submission_result,
                launch_record_path=args.launch_record,
                stdout_path=args.systemd_run_stdout,
                stderr_path=args.systemd_run_stderr,
                exit_code_path=args.systemd_run_exit_code_file,
                systemd_run_exit_code=args.systemd_run_exit_code,
            )
        elif args.command == "create-submission-receipt":
            if None in (args.receipt, args.launch_record, args.systemd_run_exit_code):
                raise OperationsError("missing_required_operational_argument")
            snapshot = None
            if args.unit_snapshot_json is not None:
                snapshot = loads_strict_object(args.unit_snapshot_json)
            elif args.unit_snapshot_file is not None:
                snapshot_payload = args.unit_snapshot_file.read_text(encoding="utf-8")
                if snapshot_payload.strip():
                    snapshot = loads_strict_object(snapshot_payload)
            value = create_submission_receipt(
                receipt_path=args.receipt,
                launch_record_path=args.launch_record,
                unit_snapshot=snapshot,
                systemd_run_exit_code=args.systemd_run_exit_code,
                observed_unit=args.unit,
                submission_result_path=args.submission_result,
                snapshot_attempt_path=args.unit_snapshot_file,
            )
        else:
            if args.run_dir is None:
                raise OperationsError("missing_required_operational_argument")
            snapshot = (
                loads_strict_object(args.unit_snapshot_json)
                if args.unit_snapshot_json is not None
                else None
            )
            value = resume_precheck(
                args.run_dir,
                launch_record_path=args.launch_record,
                unit_snapshot=snapshot,
                environment_path=args.environment,
                proposal_path=args.proposal,
                authorization_path=args.authorization,
            )
        _print(value)
        return 0 if value.get("valid", True) else 3
    except (OperationsError, StrictJSONError, OSError, json.JSONDecodeError) as exc:
        _print({"valid": False, "reason_codes": [str(exc)]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
