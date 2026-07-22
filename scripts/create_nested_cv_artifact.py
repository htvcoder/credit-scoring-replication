"""Create a non-publishable P3C nested CV/preprocessing artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from creditrep.artifacts.nested_cv import create_nested_cv_artifact
from creditrep.checksums import get_dataset_checksum
from creditrep.config.nested import load_nested_cv_config
from creditrep.datasets import load_dataset
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.nested_cv import run_nested_cv_validation
from creditrep.preprocessing.config import load_protocol_a_config
from creditrep.splitting.nested import create_nested_cv_definition, validate_nested_cv_definition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to nested CV experiment YAML config.")
    args = parser.parse_args()
    repo_root = find_repo_root()
    try:
        config = load_nested_cv_config(args.config, repo_root=repo_root)
        protocol = load_protocol_a_config(config.protocol_config_path, repo_root=repo_root)
        dataset = load_dataset(config.dataset_id, repo_root=repo_root)
        checksum = get_dataset_checksum(dataset.dataset_id, dataset.metadata["source_file"], repo_root=repo_root)
        dataset.metadata["checksum_sha256"] = checksum.actual_sha256
        nested_cv = create_nested_cv_definition(
            dataset,
            dataset_checksum=checksum.actual_sha256,
            outer_strategy=config.outer_strategy,
            outer_n_repeats=config.outer_n_repeats,
            outer_n_splits=config.outer_n_splits,
            inner_strategy=config.inner_strategy,
            inner_n_splits=config.inner_n_splits,
            shuffle=config.outer_shuffle,
            random_seed=config.outer_random_seed,
        )
        validate_nested_cv_definition(nested_cv, dataset.target)
        result = run_nested_cv_validation(
            config=config,
            dataset=dataset,
            nested_cv=nested_cv,
            protocol_config=protocol,
        )
        artifact_dir, manifest = create_nested_cv_artifact(
            config=config,
            protocol_config=protocol,
            dataset=dataset,
            checksum=checksum,
            result=result,
            repo_root=repo_root,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "dataset": manifest["dataset"]["id"],
                "outer_folds": manifest["nested_cv"]["outer_fold_count"],
                "inner_folds": manifest["nested_cv"]["inner_fold_count"],
                "nested_cv_hash": manifest["nested_cv"]["nested_cv_hash"],
                "artifact_directory": str(artifact_dir.relative_to(repo_root)).replace("\\", "/"),
                "publishable": manifest["publishable"],
                "result_scope": manifest["result_scope"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
