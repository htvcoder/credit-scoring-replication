"""Typed contracts for deterministic dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    """Result of a deterministic stratified holdout split."""

    dataset_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_target: pd.Series
    test_target: pd.Series
    metadata: dict[str, Any]
    split_hash: str
