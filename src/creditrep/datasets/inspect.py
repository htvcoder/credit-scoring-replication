"""Small CLI for inspecting a configured dataset without printing records."""

from __future__ import annotations

import argparse
import json

from creditrep.datasets.loader import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset ID, for example GC, TC or GMC.")
    args = parser.parse_args()

    loaded = load_dataset(args.dataset)
    summary = {
        "dataset_id": loaded.dataset_id,
        "file": loaded.metadata["source_file"],
        "row_count": loaded.metadata["row_count"],
        "feature_count": loaded.metadata["feature_count"],
        "class_counts": loaded.metadata["class_counts"],
        "default_rate": loaded.metadata["default_rate"],
        "removed_identifier_columns": loaded.metadata["removed_identifier_columns"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
