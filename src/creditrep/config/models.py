"""Typed P2B experiment configuration contract."""

from __future__ import annotations

from dataclasses import dataclass


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

    def canonical_payload(self) -> dict:
        return {
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
