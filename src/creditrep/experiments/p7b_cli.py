"""CLI for the non-publishable P7B CART feasibility runner."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.p7b_cart import build_plan, render_plan, run, validate_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("plan", "validate-plan", "run", "resume", "validate-artifacts"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/protocols/p7a/p7a_candidate_manifest.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = find_repo_root()
    plan = build_plan(
        args.manifest if args.manifest.is_absolute() else root / args.manifest,
        repo_root=root,
    )
    if args.command == "plan":
        render_plan(plan, args.output_dir)
        result = validate_plan(plan)
    elif args.command == "validate-plan":
        result = validate_plan(plan)
    elif args.command == "validate-artifacts":
        result = json.loads(
            (args.output_dir / "validator.json").read_text(encoding="utf-8")
        )
    else:
        result = run(
            plan, args.output_dir, repo_root=root, resume=args.command == "resume"
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
