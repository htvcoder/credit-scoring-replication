"""Experiment configuration contracts."""

from creditrep.config.loader import config_hash, load_experiment_config
from creditrep.config.models import ExperimentConfig
from creditrep.config.nested import NestedCVConfig, load_nested_cv_config, parse_nested_cv_config


def parse_model_config(*args, **kwargs):
    """Lazily import the P5 parser to avoid a package initialization cycle."""
    from creditrep.config.model_config import parse_model_config as _parse_model_config
    return _parse_model_config(*args, **kwargs)

__all__ = [
    "ExperimentConfig",
    "NestedCVConfig",
    "config_hash",
    "load_experiment_config",
    "load_nested_cv_config",
    "parse_nested_cv_config",
    "parse_model_config",
]
