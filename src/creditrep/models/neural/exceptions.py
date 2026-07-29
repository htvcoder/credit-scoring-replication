"""Classified errors for the optional neural-model foundation."""

from creditrep.models.exceptions import ModelError


class NeuralModelError(ModelError):
    """Base error for P6A neural operations."""


class NeuralDependencyError(NeuralModelError):
    """PyTorch is required but is not installed."""


class MLPConfigError(NeuralModelError):
    """The MLP configuration violates its typed contract."""


class DevicePolicyError(NeuralModelError):
    """The requested runtime device cannot be resolved."""


class NeuralInputError(NeuralModelError):
    """Training or prediction input is invalid."""


class NeuralTrainingError(NeuralModelError):
    """Training could not safely complete."""
