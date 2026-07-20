"""Load and hash portable experiment YAML configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from creditrep.config.exceptions import ConfigError
from creditrep.config.models import ExperimentConfig
from creditrep.datasets.registry import find_repo_root, validate_portable_path

SUPPORTED_SPLIT_STRATEGIES = {"stratified_holdout"}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_canonical(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def config_hash(config: ExperimentConfig) -> str:
    return sha256_canonical(config.canonical_payload())


def _as_mapping(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConfigError(f"{context}: expected a mapping.")
    return payload


def _validate_known_keys(payload: dict[str, Any]) -> None:
    allowed_top = {"experiment", "dataset", "split", "output"}
    unknown_top = sorted(set(payload) - allowed_top)
    if unknown_top:
        raise ConfigError(f"Config has unsupported top-level keys: {unknown_top}.")
    required_top = {"experiment", "dataset", "split", "output"}
    missing_top = sorted(required_top - set(payload))
    if missing_top:
        raise ConfigError(f"Config is missing required sections: {missing_top}.")

    allowed_nested = {
        "experiment": {"name"},
        "dataset": {"id"},
        "split": {"strategy", "test_size", "random_seed", "shuffle"},
        "output": {"root_dir"},
    }
    for section, allowed in allowed_nested.items():
        section_payload = _as_mapping(payload[section], context=section)
        unknown = sorted(set(section_payload) - allowed)
        if unknown:
            raise ConfigError(f"Config section {section!r} has unsupported keys: {unknown}.")


def parse_experiment_config(payload: dict[str, Any]) -> ExperimentConfig:
    _validate_known_keys(payload)
    experiment = _as_mapping(payload["experiment"], context="experiment")
    dataset = _as_mapping(payload["dataset"], context="dataset")
    split = _as_mapping(payload["split"], context="split")
    output = _as_mapping(payload["output"], context="output")

    name = experiment.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("experiment.name is required.")
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ConfigError("dataset.id is required.")

    strategy = split.get("strategy")
    if strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ConfigError(f"Unsupported split strategy {strategy!r}. Supported: {sorted(SUPPORTED_SPLIT_STRATEGIES)}.")
    test_size = split.get("test_size")
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool):
        raise ConfigError("split.test_size must be a number between 0 and 1.")
    test_size = float(test_size)
    if test_size <= 0 or test_size >= 1:
        raise ConfigError(f"split.test_size must be > 0 and < 1, got {test_size}.")
    seed = split.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("split.random_seed must be an integer.")
    shuffle = split.get("shuffle", True)
    if not isinstance(shuffle, bool):
        raise ConfigError("split.shuffle must be true or false.")

    output_root = output.get("root_dir")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ConfigError("output.root_dir is required.")
    try:
        validate_portable_path(output_root, context="output.root_dir")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    return ExperimentConfig(
        experiment_name=name.strip(),
        dataset_id=dataset_id.strip().upper(),
        split_strategy=str(strategy),
        test_size=test_size,
        random_seed=seed,
        shuffle=shuffle,
        output_root=output_root.strip(),
    )


def load_experiment_config(
    config_path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> ExperimentConfig:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise ConfigError(f"Experiment config does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Experiment config YAML is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Experiment config is malformed: {path}; expected YAML mapping.")
    return parse_experiment_config(payload)
