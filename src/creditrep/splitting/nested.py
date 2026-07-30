"""Deterministic nested cross-validation definitions for P3C."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.models import LoadedDataset
from creditrep.splitting.exceptions import SplitError
from creditrep.splitting.hashing import split_hash

NESTED_CV_VERSION = "0.1.0"


@dataclass(frozen=True)
class InnerFoldDefinition:
    parent_outer_fold_id: str
    inner_fold_id: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_class_counts: dict[int, int]
    validation_class_counts: dict[int, int]
    seed: int
    split_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_outer_fold_id": self.parent_outer_fold_id,
            "inner_fold_id": self.inner_fold_id,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "train_class_counts": {
                str(key): value for key, value in self.train_class_counts.items()
            },
            "validation_class_counts": {
                str(key): value for key, value in self.validation_class_counts.items()
            },
            "seed": self.seed,
            "split_hash": self.split_hash,
        }


@dataclass(frozen=True)
class OuterFoldDefinition:
    outer_fold_id: str
    repeat_index: int
    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_class_counts: dict[int, int]
    test_class_counts: dict[int, int]
    seed: int
    split_hash: str
    inner_folds: tuple[InnerFoldDefinition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_fold_id": self.outer_fold_id,
            "repeat_index": self.repeat_index,
            "fold_index": self.fold_index,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "train_class_counts": {
                str(key): value for key, value in self.train_class_counts.items()
            },
            "test_class_counts": {
                str(key): value for key, value in self.test_class_counts.items()
            },
            "seed": self.seed,
            "split_hash": self.split_hash,
            "inner_folds": [fold.to_dict() for fold in self.inner_folds],
        }


@dataclass(frozen=True)
class NestedCVDefinition:
    dataset_id: str
    row_count: int
    outer_strategy: str
    outer_n_repeats: int
    outer_n_splits: int
    inner_strategy: str
    inner_n_splits: int
    shuffle: bool
    base_seed: int
    dataset_checksum: str
    outer_folds: tuple[OuterFoldDefinition, ...]
    nested_cv_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_CV_VERSION,
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "outer_strategy": self.outer_strategy,
            "outer_n_repeats": self.outer_n_repeats,
            "outer_n_splits": self.outer_n_splits,
            "inner_strategy": self.inner_strategy,
            "inner_n_splits": self.inner_n_splits,
            "shuffle": self.shuffle,
            "base_seed": self.base_seed,
            "dataset_checksum": self.dataset_checksum,
            "outer_folds": [fold.to_dict() for fold in self.outer_folds],
            "nested_cv_hash": self.nested_cv_hash,
        }


def derive_seed(
    base_seed: int,
    *,
    stage: str,
    repeat_index: int,
    outer_fold_index: int,
    inner_fold_index: int | None = None,
) -> int:
    """Derive a deterministic sklearn-compatible seed without Python hash()."""

    payload = {
        "base_seed": base_seed,
        "inner_fold_index": inner_fold_index,
        "outer_fold_index": outer_fold_index,
        "repeat_index": repeat_index,
        "stage": stage,
    }
    digest = sha256_canonical(payload)
    return int(digest[:16], 16) % (2**31 - 1)


def _class_counts(target: pd.Series, indices: tuple[int, ...]) -> dict[int, int]:
    values = target.iloc[list(indices)]
    return {
        int(key): int(value)
        for key, value in values.value_counts().sort_index().items()
    }


def _validate_target(target: pd.Series, *, n_splits: int, context: str) -> None:
    if target.isna().any():
        raise SplitError(f"{context}: target contains missing values.")
    domain = set(int(value) for value in target.unique())
    if domain != {0, 1}:
        raise SplitError(
            f"{context}: target must contain exactly classes 0 and 1, got {sorted(domain)}."
        )
    counts = target.value_counts()
    min_count = int(counts.min())
    if min_count < n_splits:
        raise SplitError(
            f"{context}: minority class count {min_count} is smaller than n_splits={n_splits}."
        )


def _stratified_partitions(
    target: pd.Series,
    *,
    indices: tuple[int, ...],
    n_splits: int,
    seed: int,
    shuffle: bool,
) -> list[tuple[int, ...]]:
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for row_position in indices:
        by_class[int(target.iloc[row_position])].append(row_position)
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for klass in sorted(by_class):
        rows = list(by_class[klass])
        if shuffle:
            rng.shuffle(rows)
        for offset, row_position in enumerate(rows):
            folds[offset % n_splits].append(row_position)
    return [tuple(sorted(fold)) for fold in folds]


def _fold_hash_payload(
    *,
    dataset: LoadedDataset,
    checksum: str,
    fold_id: str,
    strategy: str,
    seed: int,
    train_indices: tuple[int, ...],
    validation_or_test_indices: tuple[int, ...],
    partition_name: str,
) -> dict[str, Any]:
    return {
        "dataset": {
            "checksum_sha256": checksum,
            "id": dataset.dataset_id,
            "source_file": dataset.metadata.get("source_file"),
        },
        "fold": {
            "fold_id": fold_id,
            "partition_name": partition_name,
            "seed": seed,
            "strategy": strategy,
            "train_indices": list(train_indices),
            partition_name: list(validation_or_test_indices),
            "target_values": [
                int(dataset.target.iloc[row])
                for row in sorted(train_indices + validation_or_test_indices)
            ],
        },
    }


def _nested_hash_payload(definition_without_hash: dict[str, Any]) -> dict[str, Any]:
    payload = dict(definition_without_hash)
    payload.pop("nested_cv_hash", None)
    return payload


def create_nested_cv_definition(
    dataset: LoadedDataset,
    *,
    dataset_checksum: str,
    outer_strategy: str = "repeated_stratified_2fold",
    outer_n_repeats: int | Mapping[str, int] = 5,
    outer_n_splits: int = 2,
    inner_strategy: str = "stratified_kfold",
    inner_n_splits: int = 5,
    shuffle: bool = True,
    random_seed: int = 42,
) -> NestedCVDefinition:
    """Create deterministic repeated 2-fold outer and stratified k-fold inner definitions."""

    if outer_strategy != "repeated_stratified_2fold":
        raise SplitError(f"Unsupported outer CV strategy {outer_strategy!r}.")
    if outer_n_splits != 2:
        raise SplitError("repeated_stratified_2fold requires outer_n_splits=2.")
    if isinstance(outer_n_repeats, Mapping):
        requested = outer_n_repeats.get(dataset.dataset_id.upper())
        if requested is None:
            raise SplitError(
                f"outer_n_repeats mapping has no entry for dataset {dataset.dataset_id.upper()}."
            )
        outer_n_repeats = requested
    if (
        not isinstance(outer_n_repeats, int)
        or isinstance(outer_n_repeats, bool)
        or outer_n_repeats < 1
    ):
        raise SplitError("outer_n_repeats must be >= 1.")
    if inner_strategy != "stratified_kfold":
        raise SplitError(f"Unsupported inner CV strategy {inner_strategy!r}.")
    if inner_n_splits < 2:
        raise SplitError("inner_n_splits must be >= 2.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise SplitError("nested CV random_seed must be an integer.")
    if len(dataset.features) != len(dataset.target):
        raise SplitError(f"{dataset.dataset_id}: features/target length mismatch.")
    _validate_target(
        dataset.target,
        n_splits=outer_n_splits,
        context=f"{dataset.dataset_id} outer CV",
    )

    all_indices = tuple(range(len(dataset.target)))
    outer_folds: list[OuterFoldDefinition] = []
    for repeat in range(outer_n_repeats):
        outer_seed = derive_seed(
            random_seed, stage="outer", repeat_index=repeat, outer_fold_index=0
        )
        test_partitions = _stratified_partitions(
            dataset.target,
            indices=all_indices,
            n_splits=2,
            seed=outer_seed,
            shuffle=shuffle,
        )
        seen_test: set[int] = set()
        for fold_index, test_indices in enumerate(test_partitions):
            outer_fold_id = f"repeat_{repeat:02d}_fold_{fold_index:02d}"
            test_set = set(test_indices)
            train_indices = tuple(row for row in all_indices if row not in test_set)
            seen_test.update(test_indices)
            outer_fold_seed = derive_seed(
                random_seed,
                stage="outer_fold",
                repeat_index=repeat,
                outer_fold_index=fold_index,
            )
            _validate_target(
                dataset.target.iloc[list(train_indices)],
                n_splits=inner_n_splits,
                context=outer_fold_id,
            )
            outer_payload = _fold_hash_payload(
                dataset=dataset,
                checksum=dataset_checksum,
                fold_id=outer_fold_id,
                strategy=outer_strategy,
                seed=outer_fold_seed,
                train_indices=train_indices,
                validation_or_test_indices=test_indices,
                partition_name="test_indices",
            )
            outer_hash = split_hash(outer_payload)
            inner_seed = derive_seed(
                random_seed,
                stage="inner",
                repeat_index=repeat,
                outer_fold_index=fold_index,
            )
            validation_partitions = _stratified_partitions(
                dataset.target,
                indices=train_indices,
                n_splits=inner_n_splits,
                seed=inner_seed,
                shuffle=shuffle,
            )
            inner_folds: list[InnerFoldDefinition] = []
            for inner_index, validation_indices in enumerate(validation_partitions):
                inner_fold_id = f"{outer_fold_id}_inner_{inner_index:02d}"
                validation_set = set(validation_indices)
                inner_train = tuple(
                    row for row in train_indices if row not in validation_set
                )
                seed = derive_seed(
                    random_seed,
                    stage="inner_fold",
                    repeat_index=repeat,
                    outer_fold_index=fold_index,
                    inner_fold_index=inner_index,
                )
                inner_payload = _fold_hash_payload(
                    dataset=dataset,
                    checksum=dataset_checksum,
                    fold_id=inner_fold_id,
                    strategy=inner_strategy,
                    seed=seed,
                    train_indices=inner_train,
                    validation_or_test_indices=validation_indices,
                    partition_name="validation_indices",
                )
                inner_folds.append(
                    InnerFoldDefinition(
                        parent_outer_fold_id=outer_fold_id,
                        inner_fold_id=inner_fold_id,
                        train_indices=inner_train,
                        validation_indices=validation_indices,
                        train_class_counts=_class_counts(dataset.target, inner_train),
                        validation_class_counts=_class_counts(
                            dataset.target, validation_indices
                        ),
                        seed=seed,
                        split_hash=split_hash(inner_payload),
                    )
                )
            if seen_test == set(all_indices) and fold_index == outer_n_splits - 1:
                pass
            outer_folds.append(
                OuterFoldDefinition(
                    outer_fold_id=outer_fold_id,
                    repeat_index=repeat,
                    fold_index=fold_index,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_class_counts=_class_counts(dataset.target, train_indices),
                    test_class_counts=_class_counts(dataset.target, test_indices),
                    seed=outer_fold_seed,
                    split_hash=outer_hash,
                    inner_folds=tuple(inner_folds),
                )
            )

    definition = NestedCVDefinition(
        dataset_id=dataset.dataset_id,
        row_count=len(dataset.target),
        outer_strategy=outer_strategy,
        outer_n_repeats=outer_n_repeats,
        outer_n_splits=outer_n_splits,
        inner_strategy=inner_strategy,
        inner_n_splits=inner_n_splits,
        shuffle=shuffle,
        base_seed=random_seed,
        dataset_checksum=dataset_checksum,
        outer_folds=tuple(outer_folds),
        nested_cv_hash="",
    )
    digest = split_hash(_nested_hash_payload(definition.to_dict()))
    return NestedCVDefinition(
        dataset_id=definition.dataset_id,
        row_count=definition.row_count,
        outer_strategy=definition.outer_strategy,
        outer_n_repeats=definition.outer_n_repeats,
        outer_n_splits=definition.outer_n_splits,
        inner_strategy=definition.inner_strategy,
        inner_n_splits=definition.inner_n_splits,
        shuffle=definition.shuffle,
        base_seed=definition.base_seed,
        dataset_checksum=definition.dataset_checksum,
        outer_folds=definition.outer_folds,
        nested_cv_hash=digest,
    )


def validate_nested_cv_definition(
    definition: NestedCVDefinition, target: pd.Series
) -> None:
    """Validate fold coverage, parent-child isolation and nested hash."""

    all_rows = set(range(len(target)))
    if definition.row_count != len(target):
        raise SplitError("Nested CV row_count does not match target length.")
    for repeat in range(definition.outer_n_repeats):
        test_seen: list[int] = []
        repeat_folds = [
            fold for fold in definition.outer_folds if fold.repeat_index == repeat
        ]
        if len(repeat_folds) != definition.outer_n_splits:
            raise SplitError(
                f"repeat_{repeat:02d}: expected {definition.outer_n_splits} outer folds."
            )
        for outer in repeat_folds:
            train = set(outer.train_indices)
            test = set(outer.test_indices)
            if train & test:
                raise SplitError(f"{outer.outer_fold_id}: outer train/test overlap.")
            if train | test != all_rows:
                raise SplitError(
                    f"{outer.outer_fold_id}: outer fold does not cover all rows."
                )
            if len(train) != len(outer.train_indices) or len(test) != len(
                outer.test_indices
            ):
                raise SplitError(
                    f"{outer.outer_fold_id}: duplicate row in outer partition."
                )
            if _class_counts(target, outer.train_indices) != outer.train_class_counts:
                raise SplitError(f"{outer.outer_fold_id}: train class counts mismatch.")
            if _class_counts(target, outer.test_indices) != outer.test_class_counts:
                raise SplitError(f"{outer.outer_fold_id}: test class counts mismatch.")
            test_seen.extend(outer.test_indices)
            validation_seen: list[int] = []
            for inner in outer.inner_folds:
                if inner.parent_outer_fold_id != outer.outer_fold_id:
                    raise SplitError(
                        f"{inner.inner_fold_id}: parent outer fold mismatch."
                    )
                inner_train = set(inner.train_indices)
                validation = set(inner.validation_indices)
                if inner_train & validation:
                    raise SplitError(
                        f"{inner.inner_fold_id}: inner train/validation overlap."
                    )
                if not inner_train <= train or not validation <= train:
                    raise SplitError(
                        f"{inner.inner_fold_id}: inner rows must be subset of outer train."
                    )
                if validation & test or inner_train & test:
                    raise SplitError(
                        f"{inner.inner_fold_id}: outer test row leaked into inner folds."
                    )
                validation_seen.extend(inner.validation_indices)
            if sorted(validation_seen) != sorted(outer.train_indices):
                raise SplitError(
                    f"{outer.outer_fold_id}: inner validation rows do not cover outer train once."
                )
        if sorted(test_seen) != sorted(all_rows):
            raise SplitError(
                f"repeat_{repeat:02d}: outer test rows do not cover all rows once."
            )
    expected = split_hash(_nested_hash_payload(definition.to_dict()))
    if expected != definition.nested_cv_hash:
        raise SplitError("Nested CV hash mismatch.")
