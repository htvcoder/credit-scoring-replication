"""CLI for P7C.4B.2c outer-refit and orchestration-overhead preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creditrep.datasets.registry import find_repo_root
from creditrep.config.loader import sha256_canonical
from creditrep.experiments.p7c4b2c_preflight import (
    EXIT_AUTHORIZATION,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_VALIDATION,
    resume,
    run,
    validate_artifacts,
)
from creditrep.experiments.p7c4b2b_preflight import (
    validate_artifacts as validate_inner_artifacts,
)
from creditrep.protocols.p7c4b2b import MODES as INNER_MODES
from creditrep.protocols.p7c4b2b import project as project_inner
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2c import (
    P7C4B2CError,
    build_plan,
    project_validated,
    validate_combined_projection_identity,
    validate_combined_projection_sources,
    validate_plan,
)
from creditrep.strict_json import StrictJSONError, load_strict_json_object


def _print(value):
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _default_plan(root: Path) -> dict:
    manifest = load_manifest(
        root / "configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml",
        repo_root=root,
    )
    return build_plan(manifest)


def _read(path: Path) -> dict:
    return load_strict_json_object(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "create-plan",
            "validate-plan",
            "run",
            "resume",
            "validate-artifacts",
            "project",
            "inspect-eligibility",
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument(
        "--execution-class",
        choices=("synthetic_validation", "target_preflight"),
        default="synthetic_validation",
    )
    parser.add_argument(
        "--mode", choices=("cpu_parallel_1", "cpu_parallel_2"), default="cpu_parallel_1"
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--target-environment", type=Path)
    parser.add_argument("--authorization-proposal", type=Path)
    parser.add_argument("--effective-authorization", type=Path)
    parser.add_argument("--inner-projection", type=Path)
    parser.add_argument("--inner-run", type=Path, action="append")
    parser.add_argument("--overhead-mapping", type=Path)
    parser.add_argument("--price-input", type=Path)
    args = parser.parse_args(argv)
    root = find_repo_root()
    plan = _default_plan(root)
    try:
        if args.command == "create-plan":
            if args.output:
                if args.output.exists():
                    raise P7C4B2CError("output_collision")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8"
                )
            _print(plan)
            return EXIT_OK
        if args.command == "validate-plan":
            _print(validate_plan(plan))
            return EXIT_OK
        if args.command == "run":
            if not args.output:
                parser.error("run requires --output")
            if args.execution_class == "target_preflight" and (
                not args.target_environment
                or not args.authorization_proposal
                or not args.effective_authorization
            ):
                raise P7C4B2CError("authorization_missing")
            result = run(
                plan,
                args.output,
                execution_class=args.execution_class,
                mode=args.mode,
                repo_root=root,
                target_environment=_read(args.target_environment)
                if args.target_environment
                else None,
                authorization_proposal=_read(args.authorization_proposal)
                if args.authorization_proposal
                else None,
                effective_authorization=_read(args.effective_authorization)
                if args.effective_authorization
                else None,
                max_samples=args.max_samples,
            )
            _print(result)
            return EXIT_OK if result["validation"]["valid"] else EXIT_VALIDATION
        if args.command == "resume":
            if not args.run_dir:
                parser.error("resume requires --run-dir")
            result = resume(
                args.run_dir[0],
                repo_root=root,
                target_environment=_read(args.target_environment)
                if args.target_environment
                else None,
                authorization_proposal=_read(args.authorization_proposal)
                if args.authorization_proposal
                else None,
                effective_authorization=_read(args.effective_authorization)
                if args.effective_authorization
                else None,
            )
            _print(result)
            return EXIT_OK if result["validation"]["valid"] else EXIT_VALIDATION
        if not args.run_dir or any(not path.exists() for path in args.run_dir):
            _print({"valid": False, "reason_codes": ["missing_required_artifact"]})
            return EXIT_VALIDATION
        reports = [validate_artifacts(path) for path in args.run_dir]
        if args.command == "validate-artifacts":
            if len(reports) != 1:
                raise P7C4B2CError("validate_artifacts_requires_one_run")
            report = reports[0]
            _print(report)
            return EXIT_OK if report["valid"] else EXIT_VALIDATION
        records = [
            _read(path)
            for run_dir in args.run_dir
            for path in sorted((run_dir / "samples").glob("*/result.json"))
        ]
        plans = [_read(path / "plan.json") for path in args.run_dir]
        manifests = [_read(path / "manifest.json") for path in args.run_dir]
        if len({item["plan_digest"] for item in plans}) != 1:
            raise P7C4B2CError("combined_plan_hash_mismatch")
        execution_classes = {item["execution_class"] for item in manifests}
        task_report = validate_combined_projection_sources(records, plans[0], reports)
        outer_entries = [
            {
                "run_directory": str(path.resolve()),
                "manifest": manifest,
                "environment": _read(path / "environment.json"),
                "validation": report,
            }
            for path, manifest, report in zip(
                args.run_dir, manifests, reports, strict=True
            )
        ]
        overhead_mapping = (
            _read(args.overhead_mapping) if args.overhead_mapping else None
        )
        inner_projection = (
            _read(args.inner_projection) if args.inner_projection else None
        )
        inner_entries = []
        if args.inner_run:
            if args.inner_projection:
                raise P7C4B2CError("duplicate_inner_projection_source")
            inner_reports = [validate_inner_artifacts(path) for path in args.inner_run]
            if any(report.get("valid") is not True for report in inner_reports):
                raise P7C4B2CError("invalid_inner_artifact_evidence")
            inner_records = [
                _read(path)
                for run_dir in args.inner_run
                for path in sorted((run_dir / "fits").glob("*/result.json"))
            ]
            inner_manifests = [
                _read(path / "run_manifest.json") for path in args.inner_run
            ]
            inner_profiles = [
                _read(path / "machine_profile.json") for path in args.inner_run
            ]
            inner_plans = [_read(path / "plan.json") for path in args.inner_run]
            inner_environments = [
                _read(path / "target_environment.json") for path in args.inner_run
            ]
            inner_proposals = [
                _read(path / "authorization_proposal.json") for path in args.inner_run
            ]
            inner_authorizations = [
                _read(path / "effective_authorization.json") for path in args.inner_run
            ]
            inner_entries = [
                {
                    "run_directory": str(path.resolve()),
                    "manifest": manifest,
                    "profile": profile,
                    "plan": inner_plan,
                    "environment": environment,
                    "proposal": proposal,
                    "authorization": authorization,
                    "validation": report,
                }
                for path, manifest, profile, inner_plan, environment, proposal, authorization, report in zip(
                    args.inner_run,
                    inner_manifests,
                    inner_profiles,
                    inner_plans,
                    inner_environments,
                    inner_proposals,
                    inner_authorizations,
                    inner_reports,
                    strict=True,
                )
            ]
            if (
                any(
                    manifest.get("evidence_scope") != "target_single_vm_measured"
                    for manifest in inner_manifests
                )
                or len(inner_manifests) != len(INNER_MODES)
                or {manifest.get("mode") for manifest in inner_manifests}
                != set(INNER_MODES)
            ):
                raise P7C4B2CError("inner_projection_not_target_evidence")
            raw_inner = project_inner(
                inner_records, evidence_scope="target_single_vm_measured"
            )
            selected_mode = (overhead_mapping or {}).get("selected_mode")
            selected = raw_inner.get(selected_mode, {})
            hours = selected.get("inner_fit_projection", {}).get(
                "conditional_work_conserving_elapsed_hours", {}
            )
            bounds = hours.get("tc_gmc_range")
            if (
                raw_inner.get("coverage", {}).get("observed_strata") != 36
                or not isinstance(bounds, list)
                or len(bounds) != 2
                or not isinstance(hours.get("point"), (int, float))
            ):
                raise P7C4B2CError("inner_projection_incomplete")
            source_hashes = sorted(
                [
                    sha256_canonical(
                        {
                            "manifest": manifest,
                            "records": [
                                record
                                for record in inner_records
                                if record.get("mode") == manifest.get("mode")
                            ],
                        }
                    )
                    for manifest in inner_manifests
                ]
            )
            inner_projection = {
                "schema_version": 1,
                "artifact_type": "p7c4b2b_validated_inner_projection",
                "valid_for_combination": True,
                "selected_mode": selected_mode,
                "conditional_elapsed_seconds": {
                    "point": float(hours["point"]) * 3600,
                    "lower": float(min(bounds)) * 3600,
                    "upper": float(max(bounds)) * 3600,
                },
                "source_evidence_digest": sha256_canonical(
                    {"source_artifact_hashes": source_hashes}
                ),
                "source_artifact_hashes": source_hashes,
            }
        identity_report = validate_combined_projection_identity(
            outer_entries, inner_entries, plans[0]
        )
        combined_report = {
            **task_report,
            "valid": task_report["valid"] and identity_report["valid"],
            "reason_codes": sorted(
                set(task_report["reason_codes"] + identity_report["reason_codes"])
            ),
            "source_git_commit": identity_report.get("source_git_commit"),
            "locked_plan_digest": plans[0].get("plan_digest"),
        }
        projection = project_validated(
            records,
            plans[0],
            artifact_validation=combined_report,
            execution_class=(
                next(iter(execution_classes))
                if len(execution_classes) == 1
                else "mixed_invalid"
            ),
            inner_projection=inner_projection,
            overhead_mapping=overhead_mapping,
            price_input=_read(args.price_input) if args.price_input else None,
        )
        _print(
            projection
            if args.command == "project"
            else {
                "execution_plan_eligible": projection["execution_plan_eligible"],
                "reason_codes": projection["reason_codes"],
            }
        )
        return EXIT_OK if projection["execution_plan_eligible"] else EXIT_INCOMPLETE
    except (P7C4B2CError, StrictJSONError) as exc:
        codes = str(exc).split(",")
        _print({"valid": False, "reason_codes": codes})
        return (
            EXIT_AUTHORIZATION
            if any("authorization" in code for code in codes)
            else EXIT_VALIDATION
        )


if __name__ == "__main__":
    raise SystemExit(main())
