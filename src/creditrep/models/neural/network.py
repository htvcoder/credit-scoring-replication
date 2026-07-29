"""Shared configurable logits-only MLP builder."""

from __future__ import annotations

from typing import Any

from creditrep.models.neural.config import MLPConfig
from creditrep.models.neural.runtime import require_torch


class ConfigurableMLP:
    """Binary MLP whose ``forward`` method returns logits, never probabilities."""

    def __new__(cls, input_dimension: int, config: MLPConfig):
        torch = require_torch()

        class _ConfigurableMLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[Any] = []
                previous = input_dimension
                for hidden in config.hidden_layers:
                    layers.append(torch.nn.Linear(previous, hidden))
                    if config.batch_normalization:
                        layers.append(torch.nn.BatchNorm1d(hidden))
                    layers.append(torch.nn.ReLU())
                    if config.dropout > 0:
                        layers.append(torch.nn.Dropout(config.dropout))
                    previous = hidden
                layers.append(torch.nn.Linear(previous, config.output_dimension))
                self.layers = torch.nn.Sequential(*layers)

            def forward(self, values):
                return self.layers(values)

        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive.")
        return _ConfigurableMLP()


def architecture_metadata(model: Any, *, input_dimension: int, config: MLPConfig) -> dict[str, Any]:
    return {
        "network": "ConfigurableMLP",
        "input_dimension": input_dimension,
        "hidden_layers": list(config.hidden_layers),
        "activation": config.activation,
        "output_dimension": config.output_dimension,
        "dropout": config.dropout,
        "batch_normalization": config.batch_normalization,
        "outputs": "logits",
        "positive_class_semantics": "P(class 1) = P(bad/default) after sigmoid at prediction boundary",
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
