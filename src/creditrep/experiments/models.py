"""Typed smoke runner results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRunResult:
    experiment_id: str
    artifact_dir: Path
    dataset_id: str
    model_type: str
    split_hash: str
    prediction_hash: str
    metrics: dict[str, Any]
    manifest: dict[str, Any]
