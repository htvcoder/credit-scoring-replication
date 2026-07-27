"""Run a reduced, non-publishable P5C model-validation experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from creditrep.checksums import get_dataset_checksum
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets import load_dataset
from creditrep.datasets.registry import find_repo_root, resolve_repo_path
from creditrep.experiments.model_validation import run_folded_model_validation
from creditrep.preprocessing import load_protocol_a_config
from creditrep.splitting.nested import create_nested_cv_definition, validate_nested_cv_definition


def main() -> int:
    parser = argparse.ArgumentParser(description="Run non-publishable P5C model validation; never scientific results.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed model/fold unit.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = find_repo_root()
    try:
        path = Path(args.config); path = path if path.is_absolute() else root / path
        with path.open(encoding="utf-8") as handle:
            config = parse_model_validation_config(yaml.safe_load(handle))
        dataset = load_dataset(config.dataset_id, repo_root=root)
        checksum = get_dataset_checksum(dataset.dataset_id, dataset.metadata["source_file"], repo_root=root)
        protocol = load_protocol_a_config(config.protocol_config_path, repo_root=root)
        nested = create_nested_cv_definition(dataset, dataset_checksum=checksum.actual_sha256, outer_n_repeats=config.outer_n_repeats, outer_n_splits=config.outer_n_splits, inner_n_splits=config.inner_n_splits, random_seed=config.random_seed)
        validate_nested_cv_definition(nested, dataset.target)
        output = resolve_repo_path(config.output_root, repo_root=root, context="output.root_dir")
        report = {"experiment": config.experiment_name, "dataset": config.dataset_id, "config_hash": config.config_hash, "nested_cv_hash": nested.nested_cv_hash, "models": sorted(config.model_candidates), "publishable": False, "result_scope": "model_validation", "validated_only": args.validate_only}
        if args.validate_only:
            print(json.dumps(report, sort_keys=True)); return 0
        artifact, summary = run_folded_model_validation(config=config, dataset=dataset, nested_cv=nested, protocol_config=protocol, output_root=output, dataset_checksum=checksum.actual_sha256, repo_root=root, resume=args.resume or config.resume, fail_fast=args.fail_fast)
        print(json.dumps(report | {"artifact": str(artifact.relative_to(root)), "completed_folds": summary["completed_fold_count"], "failed_folds": summary["failed_fold_count"], "skipped_folds": summary["resumed_skipped_fold_count"], "retried_folds": summary["retried_fold_count"]}, sort_keys=True))
        return 0 if summary["failed_fold_count"] == 0 else 2
    except Exception as exc:
        print(f"Model-validation failed: {type(exc).__name__}: {str(exc).splitlines()[0][:500]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
