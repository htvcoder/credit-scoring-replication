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
    decision_package,
    render_authorization_proposal,
    validate_authorization_proposal,
    validate_target_environment,
)

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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P7C4B2DError("invalid_json_object")
    return value


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
        ),
    )
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument(
        "--mode", choices=("cpu_parallel_1", "cpu_parallel_2"), default="cpu_parallel_1"
    )
    parser.add_argument("--output-directory")
    parser.add_argument("--stage", default="target_canary")
    args = parser.parse_args(argv)
    root = find_repo_root()
    plan = _plan(root)
    try:
        if args.command == "collect-target-environment":
            if not args.output_directory:
                raise P7C4B2DError("missing_output_directory")
            _print(
                collect_target_environment(
                    plan,
                    mode=args.mode,
                    output_directory=args.output_directory,
                    repo_root=root,
                )
            )
            return EXIT_REVIEW_BLOCKED
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
        if args.proposal is None:
            raise P7C4B2DError("authorization_proposal_invalid")
        value = validate_authorization_proposal(_read(args.proposal), plan, environment)
        _print(value)
        return EXIT_OK if value["valid"] else EXIT_INVALID
    except (P7C4B2DError, OSError, json.JSONDecodeError) as exc:
        _print({"valid": False, "reason_codes": sorted(set(str(exc).split(",")))})
        return EXIT_INVALID
    except Exception:
        _print({"valid": False, "reason_codes": ["internal_error"]})
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
