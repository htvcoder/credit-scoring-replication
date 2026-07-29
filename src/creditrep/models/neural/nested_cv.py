"""Leakage-safe deterministic early-stopping holdouts for P6C.1A."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from creditrep.config.loader import sha256_canonical
from creditrep.models.neural.exceptions import NeuralInputError


@dataclass(frozen=True)
class EarlyStoppingSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    seed: int
    split_hash: str

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": "stratified_holdout",
            "validation_fraction": 0.2,
            "seed": self.seed,
            "train_count": len(self.train_indices),
            "validation_count": len(self.validation_indices),
            "split_hash": self.split_hash,
        }


def derive_early_stopping_seed(
    *,
    experiment_seed: int,
    model_id: str,
    outer_fold_id: str,
    inner_fold_id: str,
    candidate_id: int | str,
    purpose: str = "early_stopping_split",
) -> int:
    digest = sha256_canonical(
        {
            "experiment_seed": experiment_seed,
            "model_id": model_id,
            "outer_fold_id": outer_fold_id,
            "inner_fold_id": inner_fold_id,
            "candidate_id": candidate_id,
            "purpose": purpose,
        }
    )
    return int(digest[:8], 16) % (2**31 - 1)


def create_early_stopping_split(
    indices: tuple[int, ...], y: Any, *, seed: int, validation_fraction: float = 0.2
) -> EarlyStoppingSplit:
    values = np.asarray(y)
    if (
        len(indices) != len(values)
        or len(indices) < 4
        or len(set(values)) < 2
        or min(np.bincount(values.astype(int))) < 2
    ):
        raise NeuralInputError(
            "Training partition cannot support a stratified early-stopping split."
        )
    try:
        train, valid = next(
            StratifiedShuffleSplit(
                n_splits=1, test_size=validation_fraction, random_state=seed
            ).split(np.zeros(len(values)), values)
        )
    except ValueError as exc:
        raise NeuralInputError(
            "Training partition cannot support a stratified early-stopping split."
        ) from exc
    train_indices = tuple(indices[i] for i in train)
    validation_indices = tuple(indices[i] for i in valid)
    return EarlyStoppingSplit(
        train_indices,
        validation_indices,
        seed,
        sha256_canonical(
            {"train": train_indices, "validation": validation_indices, "seed": seed}
        ),
    )
