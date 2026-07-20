"""CLI implementation for P2C smoke experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.runner import run_smoke_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    repo_root = find_repo_root()
    result = run_smoke_experiment(args.config, repo_root=repo_root)
    summary = {
        "experiment_id": result.experiment_id,
        "purpose": result.manifest["result_scope"],
        "publishable": result.manifest["publishable"],
        "dataset": result.dataset_id.upper(),
        "model": result.model_type,
        "rows": result.manifest["dataset"]["row_count"],
        "train_rows": result.manifest["split"]["train_row_count"],
        "test_rows": result.manifest["split"]["test_row_count"],
        "split_hash": result.split_hash,
        "roc_auc": result.metrics["roc_auc"],
        "accuracy": result.metrics["accuracy"],
        "f1": result.metrics["f1"],
        "prediction_hash": result.prediction_hash,
        "artifact_directory": str(result.artifact_dir.relative_to(repo_root)).replace("\\", "/"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
