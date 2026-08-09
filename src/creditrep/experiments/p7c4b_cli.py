"""CLI for P7C.4B.1 plan/readiness validation; canonical execution is unavailable here."""
import argparse, json
from pathlib import Path
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.p7c4b_mlp_benchmark import build_plan, validate_plan, run_cpu_sequential

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate-plan", "preflight", "run"))
    parser.add_argument("--mode", default="cpu_sequential")
    parser.add_argument("--plan", type=Path, default=Path("configs/protocols/p7c/p7c4a_mlp_compute_benchmark_plan.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--non-canonical-smoke", action="store_true")
    parser.add_argument("--max-fits", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args(); root = find_repo_root()
    plan = build_plan(root / args.plan, mode=args.mode, repo_root=root)
    if args.command == "run":
        if args.output_dir is None: parser.error("--output-dir is required")
        if not args.non_canonical_smoke: parser.error("B1a only authorizes --non-canonical-smoke execution")
        if args.max_fits is None or not 1 <= args.max_fits <= 3: parser.error("smoke --max-fits must be 1..3")
        result = run_cpu_sequential(plan, args.output_dir, repo_root=root, fixture=True, max_fits=args.max_fits, timeout_seconds=args.timeout_seconds)
    else: result = validate_plan(plan)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return 0 if not result.get("failed") else 2

if __name__ == "__main__": raise SystemExit(main())
