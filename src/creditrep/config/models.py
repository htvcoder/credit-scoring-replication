"""Typed P2B experiment configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated config for deterministic split artifact creation."""

    experiment_name: str
    dataset_id: str
    split_strategy: str
    test_size: float
    random_seed: int
    shuffle: bool
    output_root: str
    experiment_purpose: str | None = None
    publishable: bool | None = None
    preprocessing_mode: str | None = None
    model_type: str | None = None
    model_parameters: dict[str, Any] = field(default_factory=dict)
    classification_threshold: float | None = None

    def canonical_payload(self) -> dict:
        payload = {
            "dataset": {"id": self.dataset_id.upper()},
            "experiment": {"name": self.experiment_name},
            "output": {"root_dir": self.output_root},
            "split": {
                "random_seed": self.random_seed,
                "shuffle": self.shuffle,
                "strategy": self.split_strategy,
                "test_size": self.test_size,
            },
        }
        if self.experiment_purpose is not None:
            payload["experiment"]["purpose"] = self.experiment_purpose
        if self.publishable is not None:
            payload["experiment"]["publishable"] = self.publishable
        if self.preprocessing_mode is not None:
            payload["preprocessing"] = {"mode": self.preprocessing_mode}
        if self.model_type is not None:
            payload["model"] = {"type": self.model_type, "parameters": dict(self.model_parameters)}
        if self.classification_threshold is not None:
            payload["evaluation"] = {"classification_threshold": self.classification_threshold}
        return payload
