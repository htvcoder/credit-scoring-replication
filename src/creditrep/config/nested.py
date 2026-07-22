"""P3C nested CV validation config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from creditrep.config.exceptions import ConfigError
from creditrep.datasets.registry import find_repo_root, validate_portable_path


@dataclass(frozen=True)
class NestedCVConfig:
    experiment_name: str
    dataset_id: str
    output_root: str
    protocol_config_path: str
    outer_strategy: str
    outer_n_repeats: int
    outer_n_splits: int
    outer_shuffle: bool
    outer_random_seed: int
    inner_strategy: str
    inner_n_splits: int
    inner_shuffle: bool
    result_scope: str = "preprocessing_validation"
    publishable: bool = False
    candidates: tuple[dict[str, Any], ...] = field(default_factory=lambda: ({"name": "candidate_0"},))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "cross_validation": {
                "inner": {
                    "n_splits": self.inner_n_splits,
                    "random_seed_policy": "derived_from_outer",
                    "shuffle": self.inner_shuffle,
                    "strategy": self.inner_strategy,
                },
                "outer": {
                    "n_repeats": self.outer_n_repeats,
                    "n_splits": self.outer_n_splits,
                    "random_seed": self.outer_random_seed,
                    "shuffle": self.outer_shuffle,
                    "strategy": self.outer_strategy,
                },
            },
            "dataset": {"id": self.dataset_id.upper()},
            "experiment": {
                "name": self.experiment_name,
                "publishable": self.publishable,
                "result_scope": self.result_scope,
            },
            "output": {"root_dir": self.output_root},
            "preprocessing": {"protocol_config": self.protocol_config_path},
            "tuning": {"candidates": list(self.candidates), "purpose": "infrastructure_validation"},
        }


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context}: expected a mapping.")
    return value


def _int(value: Any, *, context: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{context} must be an integer.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{context} must be >= {minimum}.")
    return value


def _bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be true or false.")
    return value


def parse_nested_cv_config(payload: dict[str, Any]) -> NestedCVConfig:
    """Parse reduced/full nested CV validation config."""

    root = _mapping(payload, context="nested CV config")
    allowed_top = {"experiment", "dataset", "cross_validation", "preprocessing", "output", "tuning"}
    unknown = sorted(set(root) - allowed_top)
    if unknown:
        raise ConfigError(f"Nested CV config has unsupported top-level keys: {unknown}.")
    experiment = _mapping(root.get("experiment"), context="experiment")
    dataset = _mapping(root.get("dataset"), context="dataset")
    cv = _mapping(root.get("cross_validation"), context="cross_validation")
    outer = _mapping(cv.get("outer"), context="cross_validation.outer")
    inner = _mapping(cv.get("inner"), context="cross_validation.inner")
    preprocessing = _mapping(root.get("preprocessing"), context="preprocessing")
    output = _mapping(root.get("output"), context="output")
    tuning = _mapping(root.get("tuning", {"candidates": [{"name": "candidate_0"}]}), context="tuning")

    name = experiment.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("experiment.name is required.")
    result_scope = experiment.get("result_scope")
    if result_scope != "preprocessing_validation":
        raise ConfigError("P3C nested CV config must set experiment.result_scope to preprocessing_validation.")
    publishable = experiment.get("publishable")
    if publishable is not False:
        raise ConfigError("P3C nested CV config must set experiment.publishable to false.")
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ConfigError("dataset.id is required.")

    outer_strategy = outer.get("strategy")
    if outer_strategy != "repeated_stratified_2fold":
        raise ConfigError(f"Unsupported outer strategy {outer_strategy!r}.")
    outer_splits = _int(outer.get("n_splits"), context="cross_validation.outer.n_splits", minimum=2)
    if outer_splits != 2:
        raise ConfigError("repeated_stratified_2fold requires cross_validation.outer.n_splits=2.")
    outer_repeats = _int(outer.get("n_repeats"), context="cross_validation.outer.n_repeats", minimum=1)
    outer_shuffle = _bool(outer.get("shuffle"), context="cross_validation.outer.shuffle")
    seed = _int(outer.get("random_seed"), context="cross_validation.outer.random_seed")

    inner_strategy = inner.get("strategy")
    if inner_strategy != "stratified_kfold":
        raise ConfigError(f"Unsupported inner strategy {inner_strategy!r}.")
    inner_splits = _int(inner.get("n_splits"), context="cross_validation.inner.n_splits", minimum=2)
    inner_shuffle = _bool(inner.get("shuffle"), context="cross_validation.inner.shuffle")
    if inner.get("random_seed_policy") != "derived_from_outer":
        raise ConfigError("cross_validation.inner.random_seed_policy must be derived_from_outer.")

    protocol_path = preprocessing.get("protocol_config")
    if not isinstance(protocol_path, str) or not protocol_path.strip():
        raise ConfigError("preprocessing.protocol_config is required.")
    validate_portable_path(protocol_path, context="preprocessing.protocol_config")
    output_root = output.get("root_dir")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ConfigError("output.root_dir is required.")
    validate_portable_path(output_root, context="output.root_dir")

    candidates_value = tuning.get("candidates")
    if not isinstance(candidates_value, list) or not candidates_value:
        raise ConfigError("tuning.candidates must be a non-empty list.")
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates_value):
        if not isinstance(candidate, dict):
            raise ConfigError(f"tuning.candidates[{index}] must be a mapping.")
        candidates.append(dict(candidate))

    return NestedCVConfig(
        experiment_name=name.strip(),
        dataset_id=dataset_id.strip().upper(),
        output_root=output_root.strip(),
        protocol_config_path=protocol_path.strip(),
        outer_strategy=outer_strategy,
        outer_n_repeats=outer_repeats,
        outer_n_splits=outer_splits,
        outer_shuffle=outer_shuffle,
        outer_random_seed=seed,
        inner_strategy=inner_strategy,
        inner_n_splits=inner_splits,
        inner_shuffle=inner_shuffle,
        candidates=tuple(candidates),
    )


def load_nested_cv_config(config_path: Path | str, *, repo_root: Path | str | None = None) -> NestedCVConfig:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise ConfigError(f"Nested CV config does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Nested CV config YAML is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Nested CV config is malformed: {path}; expected YAML mapping.")
    return parse_nested_cv_config(payload)
