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
SUPPORTED_MODEL_TYPES = {"logistic_regression", "xgboost"}
SUPPORTED_PREPROCESSING_MODES = {"smoke_baseline"}
SMOKE_PURPOSE = "smoke_validation"
LOGISTIC_REGRESSION_PARAMETERS = {"C", "class_weight", "max_iter", "penalty", "random_state", "solver"}
XGBOOST_PARAMETERS = {
    "colsample_bytree",
    "eval_metric",
    "learning_rate",
    "max_depth",
    "n_estimators",
    "n_jobs",
    "random_state",
    "subsample",
    "tree_method",
}


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
    allowed_top = {"experiment", "dataset", "split", "output", "preprocessing", "model", "evaluation"}
    unknown_top = sorted(set(payload) - allowed_top)
    if unknown_top:
        raise ConfigError(f"Config has unsupported top-level keys: {unknown_top}.")
    required_top = {"experiment", "dataset", "split", "output"}
    missing_top = sorted(required_top - set(payload))
    if missing_top:
        raise ConfigError(f"Config is missing required sections: {missing_top}.")

    allowed_nested = {
        "experiment": {"name", "purpose", "publishable"},
        "dataset": {"id"},
        "split": {"strategy", "test_size", "random_seed", "shuffle"},
        "output": {"root_dir"},
        "preprocessing": {"mode"},
        "model": {"type", "parameters"},
        "evaluation": {"classification_threshold"},
    }
    for section, allowed in allowed_nested.items():
        if section not in payload:
            continue
        section_payload = _as_mapping(payload[section], context=section)
        unknown = sorted(set(section_payload) - allowed)
        if unknown:
            raise ConfigError(f"Config section {section!r} has unsupported keys: {unknown}.")


def _contains_nonportable_path(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.startswith("/") or "\\" in value or ".." in Path(value).parts)
    if isinstance(value, dict):
        return any(_contains_nonportable_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonportable_path(item) for item in value)
    return False


def _validate_model_parameters(model_type: str, parameters: Any, *, split_seed: int) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ConfigError("model.parameters must be a mapping.")
    try:
        canonical_json(parameters)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"model.parameters must be JSON-serializable: {exc}") from exc
    if _contains_nonportable_path(parameters):
        raise ConfigError("model.parameters must not contain absolute or non-portable local paths.")
    allowed = LOGISTIC_REGRESSION_PARAMETERS if model_type == "logistic_regression" else XGBOOST_PARAMETERS
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ConfigError(f"{model_type}: unsupported model parameters: {unknown}.")
    if "random_state" in parameters and parameters["random_state"] != split_seed:
        raise ConfigError(
            f"{model_type}: model random_state={parameters['random_state']} conflicts with split random_seed={split_seed}."
        )
    for key in ("max_iter", "n_estimators", "max_depth", "n_jobs", "random_state"):
        if key in parameters and (not isinstance(parameters[key], int) or isinstance(parameters[key], bool)):
            raise ConfigError(f"{model_type}: parameter {key} must be an integer.")
    for key in ("C", "learning_rate", "subsample", "colsample_bytree"):
        if key in parameters and (not isinstance(parameters[key], (int, float)) or isinstance(parameters[key], bool)):
            raise ConfigError(f"{model_type}: parameter {key} must be numeric.")
    if model_type == "xgboost" and parameters.get("tree_method") not in {None, "hist", "approx", "exact"}:
        raise ConfigError("xgboost: tree_method must be CPU-compatible for P2C smoke runs.")
    return dict(parameters)


def parse_experiment_config(payload: dict[str, Any]) -> ExperimentConfig:
    _validate_known_keys(payload)
    experiment = _as_mapping(payload["experiment"], context="experiment")
    dataset = _as_mapping(payload["dataset"], context="dataset")
    split = _as_mapping(payload["split"], context="split")
    output = _as_mapping(payload["output"], context="output")
    preprocessing = payload.get("preprocessing")
    model = payload.get("model")
    evaluation = payload.get("evaluation")

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

    purpose = experiment.get("purpose")
    if purpose is not None and (not isinstance(purpose, str) or not purpose.strip()):
        raise ConfigError("experiment.purpose must be a non-empty string when provided.")
    publishable = experiment.get("publishable")
    if publishable is not None and not isinstance(publishable, bool):
        raise ConfigError("experiment.publishable must be true or false when provided.")

    preprocessing_mode = None
    model_type = None
    model_parameters: dict[str, Any] = {}
    classification_threshold = None
    if preprocessing is not None or model is not None or evaluation is not None:
        if model is None:
            raise ConfigError("model.type is required for smoke experiment configs.")
        model_cfg = _as_mapping(model, context="model")
        model_type_value = model_cfg.get("type")
        if model_type_value not in SUPPORTED_MODEL_TYPES:
            raise ConfigError(
                f"Unsupported model type {model_type_value!r}. Supported: {sorted(SUPPORTED_MODEL_TYPES)}."
            )
        model_type = str(model_type_value)
        model_parameters = _validate_model_parameters(
            model_type,
            model_cfg.get("parameters", {}),
            split_seed=seed,
        )

        preprocessing_cfg = _as_mapping(preprocessing, context="preprocessing") if preprocessing is not None else {}
        preprocessing_mode_value = preprocessing_cfg.get("mode", "smoke_baseline")
        if preprocessing_mode_value not in SUPPORTED_PREPROCESSING_MODES:
            raise ConfigError(f"Unsupported preprocessing mode {preprocessing_mode_value!r}.")
        preprocessing_mode = str(preprocessing_mode_value)

        evaluation_cfg = _as_mapping(evaluation, context="evaluation") if evaluation is not None else {}
        threshold = evaluation_cfg.get("classification_threshold", 0.5)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold < 0 or threshold > 1:
            raise ConfigError(f"evaluation.classification_threshold must be in [0, 1], got {threshold!r}.")
        classification_threshold = float(threshold)

        if purpose != SMOKE_PURPOSE:
            raise ConfigError("Smoke runner configs must set experiment.purpose to smoke_validation.")
        if publishable is not False:
            raise ConfigError("Smoke runner configs must set experiment.publishable to false.")

    return ExperimentConfig(
        experiment_name=name.strip(),
        dataset_id=dataset_id.strip().upper(),
        split_strategy=str(strategy),
        test_size=test_size,
        random_seed=seed,
        shuffle=shuffle,
        output_root=output_root.strip(),
        experiment_purpose=purpose.strip() if isinstance(purpose, str) else None,
        publishable=publishable,
        preprocessing_mode=preprocessing_mode,
        model_type=model_type,
        model_parameters=model_parameters,
        classification_threshold=classification_threshold,
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
