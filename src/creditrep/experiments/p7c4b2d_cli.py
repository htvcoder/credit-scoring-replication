"""Safe, read-only P7C.4B.2d target-preflight review commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creditrep.datasets.registry import find_repo_root
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2c import build_plan
from creditrep.protocols.p7c4b2d import (
    P7C4B2DError,
    decision_package,
    render_authorization_proposal,
    validate_authorization_proposal,
    validate_target_environment,
)

EXIT_OK = 0
EXIT_REVIEW_BLOCKED = 3
EXIT_INVALID = 2


def _print(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _plan(root: Path):
    return build_plan(
        load_manifest(
            root / "configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml",
            repo_root=root,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "review-plan",
            "inspect-target-requirements",
            "render-authorization-proposal",
            "validate-authorization-proposal",
        ),
    )
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--stage", default="target_canary")
    args = parser.parse_args(argv)
    plan = _plan(find_repo_root())
    try:
        if args.command == "review-plan":
            value = decision_package(plan)
            _print(value)
            return EXIT_REVIEW_BLOCKED
        if not args.environment:
            raise P7C4B2DError("missing_target_environment_metadata")
        environment = json.loads(args.environment.read_text(encoding="utf-8"))
        if args.command == "inspect-target-requirements":
            value = validate_target_environment(environment, plan)
            _print(value)
            return EXIT_OK if value["valid"] else EXIT_REVIEW_BLOCKED
        if args.command == "render-authorization-proposal":
            task_ids = [
                item["sample_id"]
                for item in plan["tasks"]
                if item["mode"] == environment.get("execution_mode")
            ][:1]
            _print(
                render_authorization_proposal(
                    plan,
                    environment,
                    execution_stage=args.stage,
                    task_ids=task_ids,
                    expiry=None,
                )
            )
            return EXIT_OK
        if not args.proposal:
            raise P7C4B2DError("authorization_proposal_invalid")
        value = validate_authorization_proposal(
            json.loads(args.proposal.read_text(encoding="utf-8")), plan, environment
        )
        _print(value)
        return EXIT_OK if value["valid"] else EXIT_INVALID
    except (P7C4B2DError, OSError, json.JSONDecodeError) as exc:
        _print({"valid": False, "reason_codes": str(exc).split(",")})
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
