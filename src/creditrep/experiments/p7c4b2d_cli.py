"""Safe, static P7C.4B.2d evidence, review, and proposal commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creditrep.datasets.registry import find_repo_root
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2c import build_plan
from creditrep.protocols.p7c4b2d import (
    P7C4B2DError,
    collect_target_environment,
    create_effective_authorization,
    decision_package,
    render_authorization_proposal,
    validate_authorization_proposal,
    validate_effective_authorization,
    validate_target_environment,
    TARGET_EXECUTION_STAGES,
)
from creditrep.strict_json import StrictJSONError, load_strict_json_object

EXIT_OK = 0
EXIT_REVIEW_BLOCKED = 3
EXIT_INVALID = 2
EXIT_INTERNAL = 4


def _print(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _plan(root: Path) -> dict:
    return build_plan(
        load_manifest(
            root / "configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml",
            repo_root=root,
        )
    )


def _read(path: Path) -> dict:
    return load_strict_json_object(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "collect-target-environment",
            "review-plan",
            "inspect-target-requirements",
            "render-authorization-proposal",
            "validate-authorization-proposal",
            "create-effective-authorization",
            "validate-effective-authorization",
        ),
    )
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--operator-metadata", type=Path)
    parser.add_argument(
        "--mode", choices=("cpu_parallel_1", "cpu_parallel_2"), default="cpu_parallel_1"
    )
    parser.add_argument("--output-directory")
    parser.add_argument(
        "--stage", choices=sorted(TARGET_EXECUTION_STAGES), default="target_canary"
    )
    parser.add_argument("--operator-identity")
    parser.add_argument("--operator-approval")
    parser.add_argument("--expires-at")
    args = parser.parse_args(argv)
    root = find_repo_root()
    plan = _plan(root)
    try:
        if args.command == "collect-target-environment":
            if not args.output_directory:
                raise P7C4B2DError("missing_output_directory")
            metadata = _read(args.operator_metadata) if args.operator_metadata else None
            _print(
                collect_target_environment(
                    plan,
                    mode=args.mode,
                    output_directory=args.output_directory,
                    operator_metadata=metadata,
                    repo_root=root,
                )
            )
            return EXIT_REVIEW_BLOCKED
        if args.operator_metadata is not None:
            raise P7C4B2DError("operator_metadata_command_unsupported")
        environment = _read(args.environment) if args.environment else None
        if args.command == "review-plan":
            value = decision_package(
                plan,
                environment,
                repo_root=root,
                canary_complete=True,
                canary_approved=True,
            )
            _print(value)
            return (
                EXIT_OK
                if value["readiness"] == "READY_FOR_CANARY_AUTHORIZATION_REVIEW"
                else EXIT_REVIEW_BLOCKED
            )
        if environment is None:
            raise P7C4B2DError("missing_target_environment_metadata")
        if args.command == "inspect-target-requirements":
            value = validate_target_environment(environment, plan, repo_root=root)
            _print(value)
            return EXIT_OK if value["valid"] else EXIT_REVIEW_BLOCKED
        if args.command == "render-authorization-proposal":
            report = validate_target_environment(environment, plan, repo_root=root)
            if not report["valid"]:
                _print({"valid": False, "reason_codes": report["reason_codes"]})
                return EXIT_REVIEW_BLOCKED
            _print(
                render_authorization_proposal(
                    plan, environment, execution_stage=args.stage, expiry=None
                )
            )
            return EXIT_OK
        if args.command == "create-effective-authorization":
            if args.proposal is None:
                raise P7C4B2DError("authorization_proposal_invalid")
            if args.operator_identity is None:
                raise P7C4B2DError("operator_identity_missing")
            if args.operator_approval is None:
                raise P7C4B2DError("operator_approval_missing")
            if args.expires_at is None:
                raise P7C4B2DError("expiry_missing_or_invalid")
            _print(
                create_effective_authorization(
                    _read(args.proposal),
                    environment,
                    plan,
                    operator_identity=args.operator_identity,
                    operator_approval=args.operator_approval,
                    expires_at=args.expires_at,
                    repo_root=root,
                )
            )
            return EXIT_OK
        if args.command == "validate-effective-authorization":
            if args.proposal is None or args.authorization is None:
                raise P7C4B2DError("authorization_missing")
            value = validate_effective_authorization(
                _read(args.authorization),
                _read(args.proposal),
                environment,
                plan,
                repo_root=root,
            )
            _print(value)
            return EXIT_OK if value["valid"] else EXIT_INVALID
        if args.proposal is None:
            raise P7C4B2DError("authorization_proposal_invalid")
        value = validate_authorization_proposal(_read(args.proposal), plan, environment)
        _print(value)
        return EXIT_OK if value["valid"] else EXIT_INVALID
    except (P7C4B2DError, StrictJSONError) as exc:
        _print({"valid": False, "reason_codes": sorted(set(str(exc).split(",")))})
        return EXIT_INVALID
    except Exception:
        _print({"valid": False, "reason_codes": ["internal_error"]})
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
