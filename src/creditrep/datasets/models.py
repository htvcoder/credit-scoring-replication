"""Typed contracts for loaded credit-scoring datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Validated dataset registry entry."""

    dataset_id: str
    raw: dict[str, Any]
    active_file: str
    target_column: str
    target_mapping: dict[str, int]
    identifier_columns: tuple[str, ...]
    ignored_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    missing_values: tuple[Any, ...]
    reader: dict[str, Any]


@dataclass(frozen=True)
class LoadedDataset:
    """Dataset returned by the P2A loader.

    Target values are normalized to 0 = non-default/good and
    1 = default/bad. Features exclude target, identifier columns and
    ignored columns declared in the registry.
    """

    dataset_id: str
    features: pd.DataFrame
    target: pd.Series
    metadata: dict[str, Any]
    source_path: Path
