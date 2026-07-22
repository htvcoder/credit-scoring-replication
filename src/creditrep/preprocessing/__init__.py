"""Preprocessing pipelines and Protocol A contract."""

from creditrep.preprocessing.baseline import build_smoke_preprocessor
from creditrep.preprocessing.config import load_protocol_a_config, parse_protocol_a_config
from creditrep.preprocessing.exceptions import PreprocessingError
from creditrep.preprocessing.protocol import (
    UNKNOWN_CATEGORY_TOKEN,
    LeakageSafePreprocessor,
    PreprocessingProtocol,
    ProtocolAConfig,
    features_from_dataset_metadata,
)

__all__ = [
    "LeakageSafePreprocessor",
    "PreprocessingError",
    "PreprocessingProtocol",
    "ProtocolAConfig",
    "UNKNOWN_CATEGORY_TOKEN",
    "build_smoke_preprocessor",
    "features_from_dataset_metadata",
    "load_protocol_a_config",
    "parse_protocol_a_config",
]
