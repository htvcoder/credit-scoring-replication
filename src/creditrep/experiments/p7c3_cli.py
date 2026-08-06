"""CLI for the P7C.3 CPU-only MLP feasibility plan."""

import argparse
import json
from pathlib import Path
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments import p7c3_feasibility as harness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "validate-plan",
            "preflight",
            "run",
            "resume",
            "validate-artifacts",
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = find_repo_root()
    plan = harness.build_execution_plan(
        args.plan if args.plan.is_absolute() else root / args.plan, repo_root=root
    )
    if (
        args.command in {"preflight", "run", "resume", "validate-artifacts"}
        and args.output_dir is None
    ):
        parser.error("--output-dir is required")
    if args.command == "plan":
        result = harness.validate_execution_plan(plan)
    elif args.command == "validate-plan":
        result = harness.validate_execution_plan(plan)
    elif args.command == "preflight":
        result = harness.preflight(plan, args.output_dir, repo_root=root)
    elif args.command == "validate-artifacts":
        result = harness.validate_artifacts(plan, args.output_dir)
    else:
        result = harness.run(
            plan, args.output_dir, repo_root=root, resume=args.command == "resume"
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return (
        0
        if args.command != "validate-artifacts"
        or result.get("completion_status") == "completed"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
