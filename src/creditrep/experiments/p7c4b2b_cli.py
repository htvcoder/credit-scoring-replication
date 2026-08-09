"""CLI for P7C.4B.2b bounded single-VM compute preflight."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.p7c4b2b_preflight import (
    EXIT_CONFIG,
    EXIT_GUARD,
    EXIT_MISSING,
    EXIT_OK,
    EXIT_VALIDATION,
    capture_machine_profile,
    load_default_plan,
    run,
    resume,
    validate_artifacts,
)
from creditrep.protocols.p7c4b2b import (
    PreflightError,
    cost_estimate,
    project,
    proposed_execution_plan,
    ram_feasibility,
    validate_machine,
    validate_plan,
)


def _print(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "validate-plan",
            "profile-machine",
            "run",
            "resume",
            "validate-artifacts",
            "project",
            "propose-execution-plan",
        ),
    )
    parser.add_argument(
        "--mode", default="cpu_parallel_1", choices=("cpu_parallel_1", "cpu_parallel_2")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-run", type=Path, action="append")
    parser.add_argument("--price-input", type=Path)
    parser.add_argument("--two-vm-efficiency", type=float)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--machine-role", default="development_calibration_only")
    parser.add_argument("--provider", default="operator_unspecified")
    parser.add_argument("--instance-type", default="operator_unspecified")
    parser.add_argument("--target-machine-asserted", action="store_true")
    parser.add_argument("--bounded-preflight-authorized", action="store_true")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="test-only fake worker; never benchmark evidence",
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    root = find_repo_root()
    plan = load_default_plan(root)
    try:
        if args.command == "plan":
            _print(plan)
            return EXIT_OK
        if args.command == "validate-plan":
            _print(validate_plan(plan))
            return EXIT_OK
        if args.command == "profile-machine":
            profile = capture_machine_profile(
                role=args.machine_role,
                provider=args.provider,
                instance_type=args.instance_type,
                repo_root=root,
            )
            if args.profile_output:
                args.profile_output.parent.mkdir(parents=True, exist_ok=True)
                args.profile_output.write_text(
                    json.dumps(profile, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _print(profile)
            return EXIT_OK
        if args.command == "validate-artifacts":
            if not args.output_dir or not args.output_dir.exists():
                _print({"reason_codes": ["missing_provenance"]})
                return EXIT_MISSING
            report = validate_artifacts(args.output_dir)
            _print(report)
            return EXIT_OK if report["valid"] else EXIT_VALIDATION
        if args.command in {"project", "propose-execution-plan"}:
            runs = args.input_run or ([args.output_dir] if args.output_dir else [])
            if not runs or any(not item.exists() for item in runs):
                return EXIT_MISSING
            records = []
            manifests = []
            profiles = []
            for item in runs:
                report = validate_artifacts(item)
                if not report["valid"]:
                    _print(report)
                    return EXIT_VALIDATION
                manifests.append(
                    json.loads((item / "run_manifest.json").read_text(encoding="utf-8"))
                )
                profiles.append(
                    json.loads(
                        (item / "machine_profile.json").read_text(encoding="utf-8")
                    )
                )
                records.extend(
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in item.glob("fits/*/result.json")
                )
            evidence_scope = (
                "target_single_vm_measured"
                if all(
                    x.get("evidence_scope") == "target_single_vm_measured"
                    for x in manifests
                )
                else "development_fixture_non_benchmark"
            )
            price = (
                json.loads(args.price_input.read_text(encoding="utf-8"))
                if args.price_input
                else None
            )
            projection = project(
                records,
                evidence_scope=evidence_scope,
                price=price,
                two_vm_efficiency=args.two_vm_efficiency,
            )
            if projection.get("status", "").startswith("derived") and price:
                hours = projection["single_vm_parallel_2"]["derived_value"][
                    "conservative_compute_hours"
                ]
                projection["cost"] = cost_estimate(tuple(hours), price)
            if args.command == "project":
                _print(projection)
                return EXIT_OK
            if not projection.get("status", "").startswith("derived"):
                raise PreflightError("target_preflight_evidence_missing")
            evidence_digest = sha256_canonical(
                {"run_manifests": manifests, "records": records}
            )
            ram = ram_feasibility(records, profiles[0], evidence_scope=evidence_scope)
            value = proposed_execution_plan(
                git_commit=profiles[0]["git_commit"],
                preflight_plan_digest=plan["plan_digest"],
                evidence_digest=evidence_digest,
                mode=args.mode,
                runtime_range=projection.get("single_vm_parallel_2"),
                ram=ram,
                cost=projection.get("cost"),
            )
            _print(value)
            return EXIT_OK
        if not args.output_dir:
            parser.error("--output-dir is required")
        if args.fixture:
            profile = capture_machine_profile(
                role="development_calibration_only", repo_root=root
            )
        else:
            if not args.target_machine_asserted:
                raise PreflightError("target_machine_assertion_missing")
            if not args.profile:
                raise PreflightError("validated_machine_profile_missing")
            profile = json.loads(args.profile.read_text(encoding="utf-8"))
            validate_machine(profile, plan)
        function = run if args.command == "run" else resume
        result = function(
            plan,
            profile,
            args.output_dir,
            mode=args.mode,
            repo_root=root,
            fixture=args.fixture,
            bounded_authorized=args.bounded_preflight_authorized,
            max_tasks=args.max_tasks,
            timeout_seconds=args.timeout_seconds,
            fail_fast=args.fail_fast,
        )
        _print(result)
        return EXIT_OK if result["validation"]["valid"] else EXIT_VALIDATION
    except PreflightError as exc:
        _print({"error": str(exc), "reason_codes": str(exc).split(",")})
        return (
            EXIT_GUARD
            if any(
                token in str(exc)
                for token in (
                    "target",
                    "authorization",
                    "artifact_namespace",
                    "ram",
                    "disk",
                )
            )
            else EXIT_CONFIG
        )


if __name__ == "__main__":
    raise SystemExit(main())
