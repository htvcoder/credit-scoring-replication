"""Load and validate Protocol A preprocessing config files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from creditrep.datasets.registry import find_repo_root
from creditrep.preprocessing.exceptions import PreprocessingError
from creditrep.preprocessing.protocol import (
    PROTOCOL_A_NAME,
    SUPPORTED_CATEGORICAL_IMPUTATION,
    SUPPORTED_NUMERIC_IMPUTATION,
    SUPPORTED_UNSEEN_CATEGORY,
    ProtocolAConfig,
)
from creditrep.strict_yaml import StrictYAMLError, load_strict_yaml


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreprocessingError(f"{context}: expected a mapping.")
    return value


def parse_protocol_a_config(payload: dict[str, Any]) -> ProtocolAConfig:
    """Parse the narrow P3A Protocol A config schema."""

    root = _mapping(payload, context="preprocessing config")
    unknown_top = sorted(set(root) - {"protocol"})
    if unknown_top:
        raise PreprocessingError(
            f"Unsupported preprocessing config top-level keys: {unknown_top}."
        )
    protocol = _mapping(root.get("protocol"), context="protocol")
    allowed_protocol_keys = {
        "name",
        "version",
        "numeric_imputation",
        "categorical_imputation",
        "unseen_category",
        "woe",
        "vif",
        "scaling",
    }
    unknown_protocol = sorted(set(protocol) - allowed_protocol_keys)
    if unknown_protocol:
        raise PreprocessingError(
            f"Unsupported protocol config keys: {unknown_protocol}."
        )

    name = protocol.get("name")
    if name != PROTOCOL_A_NAME:
        raise PreprocessingError(f"Unsupported preprocessing protocol {name!r}.")
    numeric = _mapping(
        protocol.get("numeric_imputation"), context="protocol.numeric_imputation"
    )
    categorical = _mapping(
        protocol.get("categorical_imputation"),
        context="protocol.categorical_imputation",
    )
    unseen = _mapping(
        protocol.get("unseen_category"), context="protocol.unseen_category"
    )

    numeric_strategy = numeric.get("strategy")
    categorical_strategy = categorical.get("strategy")
    unseen_strategy = unseen.get("strategy")
    token = unseen.get("token")
    version = protocol.get("version", "p3a-v1")
    if not isinstance(version, str) or not version:
        raise PreprocessingError("protocol.version must be a non-empty string.")
    if set(numeric) != {"strategy"}:
        raise PreprocessingError(
            "protocol.numeric_imputation only supports the key 'strategy'."
        )
    if set(categorical) != {"strategy"}:
        raise PreprocessingError(
            "protocol.categorical_imputation only supports the key 'strategy'."
        )
    if set(unseen) != {"strategy", "token"}:
        raise PreprocessingError(
            "protocol.unseen_category only supports keys 'strategy' and 'token'."
        )
    if numeric_strategy not in SUPPORTED_NUMERIC_IMPUTATION:
        raise PreprocessingError(
            f"Unsupported numeric imputation strategy {numeric_strategy!r}."
        )
    if categorical_strategy not in SUPPORTED_CATEGORICAL_IMPUTATION:
        raise PreprocessingError(
            f"Unsupported categorical imputation strategy {categorical_strategy!r}."
        )
    if unseen_strategy not in SUPPORTED_UNSEEN_CATEGORY:
        raise PreprocessingError(
            f"Unsupported unseen-category strategy {unseen_strategy!r}."
        )
    if not isinstance(token, str) or not token:
        raise PreprocessingError(
            "protocol.unseen_category.token must be a non-empty string."
        )

    woe = protocol.get("woe", {"enabled": False})
    vif = protocol.get("vif", {"enabled": False})
    scaling = protocol.get("scaling", {"enabled": False, "strategy": "standard"})
    woe_cfg = _mapping(woe, context="protocol.woe")
    vif_cfg = _mapping(vif, context="protocol.vif")
    scaling_cfg = _mapping(scaling, context="protocol.scaling")

    allowed_woe = {"enabled", "scope", "smoothing", "unknown_value", "sign_convention"}
    allowed_vif = {"enabled", "threshold", "minimum_features_to_keep", "tie_break"}
    allowed_scaling = {"enabled", "strategy"}
    if sorted(set(woe_cfg) - allowed_woe):
        raise PreprocessingError(
            f"Unsupported protocol.woe keys: {sorted(set(woe_cfg) - allowed_woe)}."
        )
    if sorted(set(vif_cfg) - allowed_vif):
        raise PreprocessingError(
            f"Unsupported protocol.vif keys: {sorted(set(vif_cfg) - allowed_vif)}."
        )
    if sorted(set(scaling_cfg) - allowed_scaling):
        raise PreprocessingError(
            f"Unsupported protocol.scaling keys: {sorted(set(scaling_cfg) - allowed_scaling)}."
        )

    woe_enabled = woe_cfg.get("enabled", False)
    vif_enabled = vif_cfg.get("enabled", False)
    scaling_enabled = scaling_cfg.get("enabled", False)
    if not isinstance(woe_enabled, bool):
        raise PreprocessingError("protocol.woe.enabled must be true or false.")
    if not isinstance(vif_enabled, bool):
        raise PreprocessingError("protocol.vif.enabled must be true or false.")
    if not isinstance(scaling_enabled, bool):
        raise PreprocessingError("protocol.scaling.enabled must be true or false.")
    woe_scope = woe_cfg.get("scope", "categorical")
    if woe_scope != "categorical":
        raise PreprocessingError(f"Unsupported WOE scope {woe_scope!r}.")
    smoothing = woe_cfg.get("smoothing", 0.5)
    if (
        not isinstance(smoothing, (int, float))
        or isinstance(smoothing, bool)
        or not math.isfinite(smoothing)
        or smoothing <= 0
    ):
        raise PreprocessingError(
            f"protocol.woe.smoothing must be > 0, got {smoothing!r}."
        )
    unknown_value = woe_cfg.get("unknown_value", 0.0)
    if (
        not isinstance(unknown_value, (int, float))
        or isinstance(unknown_value, bool)
        or not math.isfinite(unknown_value)
    ):
        raise PreprocessingError("protocol.woe.unknown_value must be numeric.")
    sign = woe_cfg.get("sign_convention", "good_over_bad")
    if sign != "good_over_bad":
        raise PreprocessingError(f"Unsupported WOE sign convention {sign!r}.")
    threshold = vif_cfg.get("threshold", 10.0)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or threshold <= 1
    ):
        raise PreprocessingError(
            f"protocol.vif.threshold must be > 1, got {threshold!r}."
        )
    minimum = vif_cfg.get("minimum_features_to_keep", 1)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise PreprocessingError(
            "protocol.vif.minimum_features_to_keep must be an integer >= 1."
        )
    tie_break = vif_cfg.get("tie_break", "feature_order")
    if tie_break != "feature_order":
        raise PreprocessingError(f"Unsupported VIF tie_break policy {tie_break!r}.")
    scaling_strategy = scaling_cfg.get("strategy", "standard")
    if scaling_strategy != "standard":
        raise PreprocessingError(f"Unsupported scaling strategy {scaling_strategy!r}.")

    return ProtocolAConfig(
        protocol_name=name,
        protocol_version=version,
        numeric_imputation_strategy=numeric_strategy,
        categorical_imputation_strategy=categorical_strategy,
        unseen_category_strategy=unseen_strategy,
        unknown_token=token,
        woe_enabled=woe_enabled,
        woe_scope=woe_scope,
        woe_smoothing=float(smoothing),
        woe_unknown_value=float(unknown_value),
        woe_sign_convention=sign,
        vif_enabled=vif_enabled,
        vif_threshold=float(threshold),
        vif_minimum_features_to_keep=minimum,
        vif_tie_break=tie_break,
        scaling_enabled=scaling_enabled,
        scaling_strategy=scaling_strategy,
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
            payload = load_strict_yaml(handle)
    except StrictYAMLError as exc:
        raise PreprocessingError(
            f"Protocol A config YAML is invalid: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PreprocessingError(
            f"Protocol A config is malformed: {path}; expected YAML mapping."
        )
    return parse_protocol_a_config(payload)
