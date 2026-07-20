"""CLI implementation for creating deterministic split artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creditrep.artifacts.writer import create_split_artifact
from creditrep.checksums import get_dataset_checksum
from creditrep.config.loader import load_experiment_config
from creditrep.datasets import load_dataset
from creditrep.datasets.registry import find_repo_root
from creditrep.splitting import create_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    repo_root = find_repo_root()
    config = load_experiment_config(args.config, repo_root=repo_root)
    dataset = load_dataset(config.dataset_id, repo_root=repo_root)
    checksum = get_dataset_checksum(dataset.dataset_id, dataset.metadata["source_file"], repo_root=repo_root)
    dataset.metadata["checksum_sha256"] = checksum.actual_sha256
    split = create_split(
        dataset,
        strategy=config.split_strategy,
        test_size=config.test_size,
        random_seed=config.random_seed,
        shuffle=config.shuffle,
    )
    artifact_dir = create_split_artifact(
        config=config,
        dataset=dataset,
        split=split,
        checksum=checksum,
        repo_root=repo_root,
    )
    summary = {
        "experiment_directory": str(artifact_dir.relative_to(repo_root)).replace("\\", "/"),
        "dataset": dataset.dataset_id.upper(),
        "rows": dataset.metadata["row_count"],
        "train_rows": split.metadata["train_row_count"],
        "test_rows": split.metadata["test_row_count"],
        "train_class_counts": split.metadata["train_class_counts"],
        "test_class_counts": split.metadata["test_class_counts"],
        "seed": config.random_seed,
        "split_hash": split.split_hash,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
