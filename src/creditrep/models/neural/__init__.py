"""PyTorch MLP training foundation (P6A).

The module deliberately does not register production model IDs; that belongs to P6B.
"""

from creditrep.models.neural.config import EarlyStoppingConfig, MLPConfig, parse_mlp_config
from creditrep.models.neural.training import TrainingResult, train_mlp

__all__ = ["EarlyStoppingConfig", "MLPConfig", "TrainingResult", "parse_mlp_config", "train_mlp"]
