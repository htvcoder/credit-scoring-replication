"""Stable P6B MLP specifications; depth is paper-exact, defaults are project decisions."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
from creditrep.models.neural.config import EarlyStoppingConfig, MLPConfig
from creditrep.models.neural.exceptions import MLPConfigError

FAIR_BUDGET_ID = "p6b_shared_v1"


@dataclass(frozen=True)
class MLPModelSpecification:
    model_id: str
    display_name: str
    hidden_layers: tuple[int, ...]
    architecture_source: str = "paper_exact"
    replication_role: str = "replication_model"
    framework: str = "pytorch"

    def __post_init__(self):
        if self.model_id not in {"mlp_1", "mlp_3", "mlp_5"} or len(
            self.hidden_layers
        ) != int(self.model_id[-1]):
            raise MLPConfigError("MLP model ID must match its hidden-layer depth.")

    def config(self, **overrides: Any) -> MLPConfig:
        patience = overrides.pop("early_stopping_patience", 20)
        min_delta = overrides.pop("early_stopping_min_delta", 0.0001)
        values = dict(
            model_id=self.model_id,
            hidden_layers=self.hidden_layers,
            random_seed=42,
            activation="relu",
            dropout=0.0,
            batch_normalization=False,
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=0.0,
            batch_size=32,
            max_epochs=200,
            early_stopping=EarlyStoppingConfig(True, patience, min_delta),
            device_policy="auto",
            dataloader_workers=0,
            checkpoint_policy="none",
            save_model=False,
            publishable=False,
            result_scope="mlp_model_validation",
        )
        values.update(overrides)
        cfg = MLPConfig(**values)
        if len(cfg.hidden_layers) != len(self.hidden_layers):
            raise MLPConfigError(
                f"{self.model_id} requires exactly {len(self.hidden_layers)} hidden layers."
            )
        return cfg

    def to_dict(self):
        return asdict(self) | {
            "hidden_depth": len(self.hidden_layers),
            "training_budget_id": FAIR_BUDGET_ID,
            "provenance": {
                "hidden_depth": "paper_exact",
                "hidden_width": "project_decision",
                "training_defaults": "project_decision",
            },
            "output_contract": "logits",
            "probability_semantics": "P(class 1) = P(bad/default)",
            "publishable": False,
            "result_scope": "mlp_model_validation",
        }


MLP_SPECS = {
    "mlp_1": MLPModelSpecification("mlp_1", "MLP-1", (64,)),
    "mlp_3": MLPModelSpecification("mlp_3", "MLP-3", (64, 64, 64)),
    "mlp_5": MLPModelSpecification("mlp_5", "MLP-5", (64, 64, 64, 64, 64)),
}


def get_mlp_specification(model_id: str) -> MLPModelSpecification:
    try:
        return MLP_SPECS[model_id]
    except KeyError as exc:
        raise MLPConfigError(f"Unknown MLP model ID: {model_id!r}.") from exc
