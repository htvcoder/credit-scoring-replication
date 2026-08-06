"""CLI for the immutable non-publishable P7C.2 RF/XGBoost pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.p7c2_feasibility import (
    EXPECTED_OUTER,
    P7C2HarnessError,
    build_execution_plan,
    run,
    validate_artifacts,
    validate_execution_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "validate-plan", "run", "resume", "validate-artifacts"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/protocols/p7c/p7c2_rf_xgboost_pilot_plan.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = find_repo_root()
    plan = build_execution_plan(
        args.plan if args.plan.is_absolute() else root / args.plan, repo_root=root
    )
    if (
        args.command in {"run", "resume", "validate-artifacts"}
        and args.output_dir is None
    ):
        parser.error("--output-dir is required for run/resume/validate-artifacts")
    if args.command == "plan":
        result = {
            "valid": True,
            "models": ["random_forest", "xgboost"],
            "datasets": ["AC", "GMC"],
            "outer_partition": EXPECTED_OUTER,
            "expected_fits": len(plan["fits"]),
            "plan_digest": plan["plan_digest"],
        }
    elif args.command == "validate-plan":
        result = validate_execution_plan(plan)
    elif args.command == "validate-artifacts":
        result = validate_artifacts(plan, args.output_dir)
    else:
        result = run(
            plan, args.output_dir, repo_root=root, resume=args.command == "resume"
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.command == "validate-artifacts":
        if result.get("valid") and result.get("completion_status") == "completed":
            return 0
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P7C2HarnessError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
