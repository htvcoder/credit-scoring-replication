"""Load and validate Protocol A preprocessing config files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from creditrep.datasets.registry import find_repo_root
from creditrep.preprocessing.exceptions import PreprocessingError
from creditrep.preprocessing.protocol import (
    PROTOCOL_A_NAME,
    SUPPORTED_CATEGORICAL_IMPUTATION,
    SUPPORTED_NUMERIC_IMPUTATION,
    SUPPORTED_UNSEEN_CATEGORY,
    ProtocolAConfig,
)


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreprocessingError(f"{context}: expected a mapping.")
    return value


def parse_protocol_a_config(payload: dict[str, Any]) -> ProtocolAConfig:
    """Parse the narrow P3A Protocol A config schema."""

    root = _mapping(payload, context="preprocessing config")
    unknown_top = sorted(set(root) - {"protocol"})
    if unknown_top:
        raise PreprocessingError(f"Unsupported preprocessing config top-level keys: {unknown_top}.")
    protocol = _mapping(root.get("protocol"), context="protocol")
    allowed_protocol_keys = {"name", "numeric_imputation", "categorical_imputation", "unseen_category"}
    unknown_protocol = sorted(set(protocol) - allowed_protocol_keys)
    if unknown_protocol:
        raise PreprocessingError(f"Unsupported protocol config keys: {unknown_protocol}.")

    name = protocol.get("name")
    if name != PROTOCOL_A_NAME:
        raise PreprocessingError(f"Unsupported preprocessing protocol {name!r}.")
    numeric = _mapping(protocol.get("numeric_imputation"), context="protocol.numeric_imputation")
    categorical = _mapping(protocol.get("categorical_imputation"), context="protocol.categorical_imputation")
    unseen = _mapping(protocol.get("unseen_category"), context="protocol.unseen_category")

    numeric_strategy = numeric.get("strategy")
    categorical_strategy = categorical.get("strategy")
    unseen_strategy = unseen.get("strategy")
    token = unseen.get("token")
    if set(numeric) != {"strategy"}:
        raise PreprocessingError("protocol.numeric_imputation only supports the key 'strategy'.")
    if set(categorical) != {"strategy"}:
        raise PreprocessingError("protocol.categorical_imputation only supports the key 'strategy'.")
    if set(unseen) != {"strategy", "token"}:
        raise PreprocessingError("protocol.unseen_category only supports keys 'strategy' and 'token'.")
    if numeric_strategy not in SUPPORTED_NUMERIC_IMPUTATION:
        raise PreprocessingError(f"Unsupported numeric imputation strategy {numeric_strategy!r}.")
    if categorical_strategy not in SUPPORTED_CATEGORICAL_IMPUTATION:
        raise PreprocessingError(f"Unsupported categorical imputation strategy {categorical_strategy!r}.")
    if unseen_strategy not in SUPPORTED_UNSEEN_CATEGORY:
        raise PreprocessingError(f"Unsupported unseen-category strategy {unseen_strategy!r}.")
    if not isinstance(token, str) or not token:
        raise PreprocessingError("protocol.unseen_category.token must be a non-empty string.")

    return ProtocolAConfig(
        protocol_name=name,
        numeric_imputation_strategy=numeric_strategy,
        categorical_imputation_strategy=categorical_strategy,
        unseen_category_strategy=unseen_strategy,
        unknown_token=token,
    )


def load_protocol_a_config(
    config_path: Path | str = "configs/protocols/protocol_a.yaml",
    *,
    repo_root: Path | str | None = None,
) -> ProtocolAConfig:
    """Load the Protocol A YAML config from a portable repository path."""

    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise PreprocessingError(f"Protocol A config does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise PreprocessingError(f"Protocol A config YAML is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreprocessingError(f"Protocol A config is malformed: {path}; expected YAML mapping.")
    return parse_protocol_a_config(payload)
