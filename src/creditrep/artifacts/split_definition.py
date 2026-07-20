"""Read, write and validate compact split definition artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.splitting.hashing import split_hash as compute_split_hash


def write_split_csv(path: Path, *, train_indices: tuple[int, ...], test_indices: tuple[int, ...]) -> None:
    assignments = {row: "train" for row in train_indices}
    assignments.update({row: "test" for row in test_indices})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_position", "partition"])
        writer.writeheader()
        for row_position in sorted(assignments):
            writer.writerow({"row_position": row_position, "partition": assignments[row_position]})


def load_split_csv(path: Path) -> dict[int, str]:
    assignments: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row_position", "partition"]:
            raise ArtifactError(f"Split definition must have row_position,partition header: {path}")
        for row in reader:
            try:
                row_position = int(row["row_position"])
            except (TypeError, ValueError) as exc:
                raise ArtifactError(f"Invalid row_position in split definition: {row}") from exc
            partition = row["partition"]
            if partition not in {"train", "test"}:
                raise ArtifactError(f"Unknown partition {partition!r} for row_position {row_position}.")
            if row_position in assignments:
                raise ArtifactError(f"Duplicate row_position in split definition: {row_position}.")
            assignments[row_position] = partition
    return assignments


def validate_split_definition(
    path: Path,
    *,
    row_count: int,
    expected_split_hash: str,
    split_hash_payload: dict,
) -> None:
    assignments = load_split_csv(path)
    expected_rows = set(range(row_count))
    actual_rows = set(assignments)
    if actual_rows != expected_rows:
        missing = sorted(expected_rows - actual_rows)[:10]
        extra = sorted(actual_rows - expected_rows)[:10]
        raise ArtifactError(f"Split definition rows mismatch; missing={missing}, extra={extra}.")
    train = tuple(sorted(row for row, partition in assignments.items() if partition == "train"))
    test = tuple(sorted(row for row, partition in assignments.items() if partition == "test"))
    payload = dict(split_hash_payload)
    payload["split"] = dict(payload["split"])
    payload["split"]["train_indices"] = list(train)
    payload["split"]["test_indices"] = list(test)
    actual_hash = compute_split_hash(payload)
    if actual_hash != expected_split_hash:
        raise ArtifactError(
            f"Split hash mismatch for {path}; expected {expected_split_hash}, actual {actual_hash}."
        )
