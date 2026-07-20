"""Deterministic stratified holdout splitter."""

from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd

from creditrep.datasets.models import LoadedDataset
from creditrep.splitting.exceptions import SplitError
from creditrep.splitting.hashing import split_hash as compute_split_hash
from creditrep.splitting.models import SplitResult


def _class_counts(series: pd.Series) -> dict[int, int]:
    return {int(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _validate_inputs(dataset: LoadedDataset, *, strategy: str, test_size: float, random_seed: int) -> None:
    if strategy != "stratified_holdout":
        raise SplitError(f"{dataset.dataset_id}: unsupported split strategy {strategy!r}.")
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or test_size <= 0 or test_size >= 1:
        raise SplitError(f"{dataset.dataset_id}: test_size must be > 0 and < 1, got {test_size!r}.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise SplitError(f"{dataset.dataset_id}: random_seed must be an integer.")
    if len(dataset.features) == 0 or len(dataset.target) == 0:
        raise SplitError(f"{dataset.dataset_id}: dataset is empty.")
    if len(dataset.features) != len(dataset.target):
        raise SplitError(
            f"{dataset.dataset_id}: features and target length mismatch: "
            f"{len(dataset.features)} != {len(dataset.target)}."
        )
    if dataset.features.index.has_duplicates or dataset.target.index.has_duplicates:
        raise SplitError(f"{dataset.dataset_id}: duplicate indices are not allowed for split provenance.")
    if not dataset.features.index.equals(dataset.target.index):
        raise SplitError(f"{dataset.dataset_id}: features and target indices must match.")
    domain = set(int(value) for value in dataset.target.unique())
    if domain != {0, 1}:
        raise SplitError(f"{dataset.dataset_id}: target must contain exactly classes 0 and 1, got {sorted(domain)}.")
    counts = _class_counts(dataset.target)
    for klass, count in counts.items():
        if count < 2:
            raise SplitError(f"{dataset.dataset_id}: class {klass} has too few rows for stratified split: {count}.")
        test_count = int(round(count * test_size))
        if test_count < 1 or test_count >= count:
            raise SplitError(
                f"{dataset.dataset_id}: test_size={test_size} leaves class {klass} without train/test coverage."
            )


def _positions_by_class(target: pd.Series) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {0: [], 1: []}
    for row_position, value in enumerate(target.tolist()):
        positions[int(value)].append(row_position)
    return positions


def _split_positions(
    target: pd.Series,
    *,
    test_size: float,
    random_seed: int,
    shuffle: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rng = random.Random(random_seed)
    train_positions: list[int] = []
    test_positions: list[int] = []
    for klass in sorted(_positions_by_class(target)):
        positions = _positions_by_class(target)[klass]
        if shuffle:
            rng.shuffle(positions)
        test_count = int(round(len(positions) * test_size))
        if not math.isfinite(test_count):
            raise SplitError("Invalid test row count.")
        test_positions.extend(positions[:test_count])
        train_positions.extend(positions[test_count:])
    train = tuple(sorted(train_positions))
    test = tuple(sorted(test_positions))
    if set(train) & set(test):
        raise SplitError("Split created overlapping train/test rows.")
    all_positions = set(range(len(target)))
    if set(train) | set(test) != all_positions:
        raise SplitError("Split lost or added rows.")
    return train, test


def _hash_payload(
    *,
    dataset: LoadedDataset,
    strategy: str,
    test_size: float,
    random_seed: int,
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "dataset": {
            "id": dataset.dataset_id,
            "checksum_sha256": dataset.metadata.get("checksum_sha256"),
            "source_file": dataset.metadata.get("source_file"),
        },
        "split": {
            "strategy": strategy,
            "test_size": test_size,
            "random_seed": random_seed,
            "train_indices": list(train_indices),
            "test_indices": list(test_indices),
        },
    }


def create_split(
    dataset: LoadedDataset,
    *,
    strategy: str = "stratified_holdout",
    test_size: float = 0.2,
    random_seed: int = 42,
    shuffle: bool = True,
) -> SplitResult:
    """Create a deterministic stratified holdout split using row positions."""

    _validate_inputs(dataset, strategy=strategy, test_size=test_size, random_seed=random_seed)
    train_indices, test_indices = _split_positions(
        dataset.target,
        test_size=test_size,
        random_seed=random_seed,
        shuffle=shuffle,
    )
    train_features = dataset.features.iloc[list(train_indices)]
    test_features = dataset.features.iloc[list(test_indices)]
    train_target = dataset.target.iloc[list(train_indices)]
    test_target = dataset.target.iloc[list(test_indices)]
    payload = _hash_payload(
        dataset=dataset,
        strategy=strategy,
        test_size=float(test_size),
        random_seed=random_seed,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    digest = compute_split_hash(payload)
    metadata = {
        "dataset_id": dataset.dataset_id,
        "strategy": strategy,
        "test_size": float(test_size),
        "random_seed": random_seed,
        "shuffle": shuffle,
        "row_index_contract": "row_position",
        "row_count": int(len(dataset.target)),
        "train_row_count": int(len(train_indices)),
        "test_row_count": int(len(test_indices)),
        "train_class_counts": _class_counts(train_target),
        "test_class_counts": _class_counts(test_target),
        "split_hash": digest,
    }
    return SplitResult(
        dataset_id=dataset.dataset_id,
        train_indices=train_indices,
        test_indices=test_indices,
        train_features=train_features,
        test_features=test_features,
        train_target=train_target,
        test_target=test_target,
        metadata=metadata,
        split_hash=digest,
    )
