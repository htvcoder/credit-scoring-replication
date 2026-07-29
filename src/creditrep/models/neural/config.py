"""Typed and JSON-safe configuration contracts for the P6A MLP foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from creditrep.config.loader import canonical_json
from creditrep.models.neural.exceptions import MLPConfigError
from creditrep.models.neural.runtime import DevicePolicy


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.enabled and (not isinstance(self.patience, int) or isinstance(self.patience, bool) or self.patience <= 0):
            raise MLPConfigError("early_stopping.patience must be a positive integer when enabled.")
        if not isinstance(self.min_delta, (int, float)) or isinstance(self.min_delta, bool) or self.min_delta < 0:
            raise MLPConfigError("early_stopping.min_delta must be a non-negative number.")


@dataclass(frozen=True)
class MLPConfig:
    hidden_layers: tuple[int, ...]
    random_seed: int
    model_id: str = "mlp_foundation"
    input_dimension: int | None = None
    activation: Literal["relu"] = "relu"
    output_dimension: int = 1
    dropout: float = 0.0
    batch_normalization: bool = False
    optimizer: Literal["adam", "adamw"] = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 20
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    gradient_clip_norm: float | None = None
    device_policy: DevicePolicy = "auto"
    deterministic: bool = True
    dataloader_workers: int = 0
    checkpoint_policy: Literal["none"] = "none"
    save_model: bool = False
    publishable: bool = False
    result_scope: str = "mlp_training_validation"

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            raise MLPConfigError("hidden_layers must not be empty.")
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.hidden_layers):
            raise MLPConfigError("Every hidden layer size must be a positive integer.")
        if self.input_dimension is not None and (not isinstance(self.input_dimension, int) or self.input_dimension <= 0):
            raise MLPConfigError("input_dimension must be a positive integer when supplied.")
        if self.output_dimension != 1:
            raise MLPConfigError("output_dimension must be 1 for binary classification logits.")
        if self.activation != "relu":
            raise MLPConfigError(f"Unsupported activation: {self.activation!r}.")
        if self.optimizer not in {"adam", "adamw"}:
            raise MLPConfigError(f"Unsupported optimizer: {self.optimizer!r}.")
        if self.device_policy not in {"auto", "cpu", "cuda"}:
            raise MLPConfigError(f"Unsupported device policy: {self.device_policy!r}.")
        if not 0 <= self.dropout < 1:
            raise MLPConfigError("dropout must be in [0, 1).")
        for name, value, allow_zero in (("learning_rate", self.learning_rate, False), ("weight_decay", self.weight_decay, True)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or (not allow_zero and value == 0):
                raise MLPConfigError(f"{name} must be {'non-negative' if allow_zero else 'positive'}.")
        for name, value in (("batch_size", self.batch_size), ("max_epochs", self.max_epochs)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MLPConfigError(f"{name} must be a positive integer.")
        if self.gradient_clip_norm is not None and (not isinstance(self.gradient_clip_norm, (int, float)) or self.gradient_clip_norm <= 0):
            raise MLPConfigError("gradient_clip_norm must be positive when supplied.")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise MLPConfigError("random_seed must be an integer.")
        if not isinstance(self.dataloader_workers, int) or isinstance(self.dataloader_workers, bool) or self.dataloader_workers < 0:
            raise MLPConfigError("dataloader_workers must be a non-negative integer.")
        if self.checkpoint_policy != "none" or self.save_model:
            raise MLPConfigError("P6A only supports in-memory best-weight snapshots; disk checkpoints are disabled.")
        if self.publishable or self.result_scope not in {"mlp_training_validation", "mlp_model_validation"}:
            raise MLPConfigError("Neural results must be non-publishable with an approved validation scope.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_layers"] = list(self.hidden_layers)
        canonical_json(payload)
        return payload


def parse_mlp_config(payload: Mapping[str, Any]) -> MLPConfig:
    allowed = {"model_id", "input_dimension", "hidden_layers", "activation", "output_dimension", "dropout", "batch_normalization", "optimizer", "learning_rate", "weight_decay", "batch_size", "max_epochs", "early_stopping", "gradient_clip_norm", "random_seed", "device_policy", "deterministic", "dataloader_workers", "checkpoint_policy", "save_model", "publishable", "result_scope"}
    if not isinstance(payload, Mapping):
        raise MLPConfigError("MLP configuration must be a mapping.")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise MLPConfigError(f"MLP configuration has unsupported keys: {unknown}.")
    early = payload.get("early_stopping", {})
    if not isinstance(early, Mapping):
        raise MLPConfigError("early_stopping must be a mapping.")
    try:
        early_config = EarlyStoppingConfig(**dict(early))
        return MLPConfig(**(dict(payload) | {"hidden_layers": tuple(payload.get("hidden_layers", ())), "early_stopping": early_config}))
    except TypeError as exc:
        raise MLPConfigError(str(exc)) from exc
