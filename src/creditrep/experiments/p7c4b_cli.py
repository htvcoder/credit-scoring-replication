"""P7C.4B.1b artifact validation and bounded CPU-sequential resume CLI."""
import argparse
import json
from pathlib import Path

from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.p7c4b_mlp_benchmark import (
    EXIT_CORRUPT, EXIT_INCOMPATIBLE_RESUME, EXIT_INTERRUPTED, EXIT_INVALID_CONFIG,
    EXIT_MISSING_RUN, EXIT_VALID, EXIT_VALIDATION_FAILURE, P7C4BBenchmarkError,
    build_plan, quarantine_corrupt, resume_cpu_parallel_2, resume_cpu_sequential, validate_artifacts,
    validate_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate-plan", "preflight", "run", "resume", "validate-artifacts", "quarantine"))
    parser.add_argument("--mode", default="cpu_sequential")
    parser.add_argument("--plan", type=Path, default=Path("configs/protocols/p7c/p7c4a_mlp_compute_benchmark_plan.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--non-canonical-smoke", action="store_true")
    parser.add_argument("--max-fits", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--stop-after", type=int, help="test-only bounded interruption after N newly executed fits")
    args = parser.parse_args(); root = find_repo_root()
    if args.mode not in {"cpu_sequential", "cpu_parallel_2"}:
        print(json.dumps({"error": "unsupported_mode"})); return EXIT_INVALID_CONFIG
    if args.command in {"validate-artifacts", "quarantine"}:
        if args.output_dir is None: parser.error("--output-dir is required")
        if not args.output_dir.exists(): print(json.dumps({"reason_codes": ["missing_manifest"]})); return EXIT_MISSING_RUN
        report = validate_artifacts(args.output_dir, write_report=args.command == "validate-artifacts")
        if args.command == "quarantine": report["quarantine"] = str(quarantine_corrupt(args.output_dir, report) or "")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return EXIT_VALID if report["valid"] else EXIT_CORRUPT if "corrupt_artifact" in report["reason_codes"] else EXIT_VALIDATION_FAILURE
    try:
        plan = build_plan(root / args.plan, mode=args.mode, repo_root=root)
        if args.command in {"run", "resume"}:
            if args.output_dir is None or not args.non_canonical_smoke: parser.error("--output-dir and --non-canonical-smoke are required")
            runner = resume_cpu_parallel_2 if args.mode == "cpu_parallel_2" else resume_cpu_sequential
            result = runner(plan, args.output_dir, repo_root=root, max_fits=args.max_fits, timeout_seconds=args.timeout_seconds, stop_after=args.stop_after)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return EXIT_INTERRUPTED if args.stop_after is not None and result["executed"] >= args.stop_after else EXIT_VALID
        result = validate_plan(plan)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return EXIT_VALID
    except P7C4BBenchmarkError as exc:
        message = str(exc); print(json.dumps({"error": message}))
        return EXIT_INCOMPATIBLE_RESUME if "incompatible" in message else EXIT_CORRUPT if "corrupt" in message else EXIT_INVALID_CONFIG


if __name__ == "__main__": raise SystemExit(main())
