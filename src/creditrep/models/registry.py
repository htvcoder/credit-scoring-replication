"""Single source of truth for stable Phase 5 model identifiers."""

from __future__ import annotations

import math
from typing import Any, Mapping

from creditrep.config.exceptions import ConfigError
from creditrep.models.contract import ModelCapability, ModelConfig
from creditrep.models.exceptions import ModelError


class ModelRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, ModelCapability] = {}

    def register(self, capability: ModelCapability) -> None:
        if capability.model_id in self._capabilities:
            raise ModelError(f"Duplicate model registration: {capability.model_id}.")
        self._capabilities[capability.model_id] = capability

    def resolve(self, model_id: str) -> ModelCapability:
        try:
            return self._capabilities[model_id]
        except KeyError as exc:
            raise ModelError(f"Unknown model ID: {model_id!r}.") from exc

    def validate_config(self, config: ModelConfig) -> ModelCapability:
        try:
            capability = self.resolve(config.model_id)
        except ModelError as exc:
            raise ConfigError(str(exc)) from exc
        validate_hyperparameters(config.model_id, config.hyperparameters)
        return capability

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return tuple(self._capabilities[key] for key in sorted(self._capabilities))


MODEL_REGISTRY = ModelRegistry()

MODEL_REGISTRY.register(
    ModelCapability(
        "logistic_regression",
        "Logistic Regression",
        "linear",
        "LogisticRegression",
        "scikit-learn",
        True,
        True,
        allowed_hyperparameters=(
            "C",
            "class_weight",
            "max_iter",
            "penalty",
            "random_state",
            "solver",
        ),
        default_hyperparameters={
            "max_iter": 200,
            "solver": "liblinear",
            "random_state": 42,
        },
        algorithm="logistic_regression",
        implementation="sklearn.linear_model.LogisticRegression",
        implemented=True,
    )
)
MODEL_REGISTRY.register(
    ModelCapability(
        "decision_tree",
        "Decision Tree (CART)",
        "tree",
        "DecisionTreeClassifier",
        "scikit-learn",
        True,
        True,
        allowed_hyperparameters=(
            "criterion",
            "splitter",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "random_state",
            "class_weight",
            "ccp_alpha",
        ),
        default_hyperparameters={
            "criterion": "gini",
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "class_weight": None,
        },
        algorithm="cart",
        implementation="sklearn.tree.DecisionTreeClassifier",
        replication_role="approximation",
        deviation_from_paper="c45_to_cart",
        implemented=True,
    )
)
MODEL_REGISTRY.register(
    ModelCapability(
        "random_forest",
        "Random Forest",
        "ensemble",
        "RandomForestClassifier",
        "scikit-learn",
        True,
        True,
        allowed_hyperparameters=(
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "class_weight",
            "n_jobs",
            "random_state",
        ),
        default_hyperparameters={
            "n_estimators": 20,
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": 1,
            "random_state": 42,
            "class_weight": None,
        },
        algorithm="random_forest",
        implementation="sklearn.ensemble.RandomForestClassifier",
        implemented=True,
    )
)
MODEL_REGISTRY.register(
    ModelCapability(
        "xgboost",
        "XGBoost",
        "gradient_boosting",
        "XGBClassifier",
        "xgboost",
        True,
        True,
        allowed_hyperparameters=(
            "colsample_bytree",
            "eval_metric",
            "learning_rate",
            "max_depth",
            "n_estimators",
            "n_jobs",
            "random_state",
            "subsample",
            "tree_method",
        ),
        default_hyperparameters={
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "n_jobs": 1,
            "random_state": 42,
            "eval_metric": "logloss",
            "tree_method": "hist",
        },
        algorithm="gradient_boosted_trees",
        implementation="xgboost.XGBClassifier",
        implemented=True,
    )
)
for _id, _depth in (("mlp_1", 1), ("mlp_3", 3), ("mlp_5", 5)):
    MODEL_REGISTRY.register(
        ModelCapability(
            _id,
            f"MLP-{_depth}",
            "neural_network",
            "MLPProbabilityEstimator",
            "pytorch",
            True,
            True,
            allowed_hyperparameters=(
                "batch_size",
                "batch_normalization",
                "device_policy",
                "dropout",
                "early_stopping_min_delta",
                "early_stopping_patience",
                "hidden_layers",
                "learning_rate",
                "max_epochs",
                "optimizer",
                "random_seed",
                "weight_decay",
            ),
            default_hyperparameters={
                "random_seed": 42,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "batch_size": 32,
                "max_epochs": 200,
                "device_policy": "auto",
            },
            algorithm="mlp",
            implementation="creditrep.models.neural.MLPProbabilityEstimator",
            replication_role="replication_model",
            deviation_from_paper="training_defaults_project_decision",
            implemented=True,
        )
    )


def validate_hyperparameters(model_id: str, parameters: Mapping[str, Any]) -> None:
    capability = MODEL_REGISTRY.resolve(model_id)
    unknown = sorted(set(parameters) - set(capability.allowed_hyperparameters))
    if unknown:
        raise ConfigError(f"{model_id}: unknown hyperparameters: {unknown}.")
    positive_ints = {
        "max_iter",
        "max_depth",
        "min_samples_split",
        "n_estimators",
    }
    for name, value in parameters.items():
        if name in positive_ints and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ConfigError(
                f"{model_id}: hyperparameter {name} must be a positive integer."
            )
        if name == "min_samples_leaf":
            if isinstance(value, bool):
                raise ConfigError(f"{model_id}: min_samples_leaf must not be boolean.")
            if isinstance(value, int):
                if value <= 0:
                    raise ConfigError(f"{model_id}: min_samples_leaf integer must be positive.")
            elif isinstance(value, float):
                if not math.isfinite(value) or not 0 < value <= 0.5:
                    raise ConfigError(
                        f"{model_id}: min_samples_leaf fraction must be finite and in (0, 0.5]."
                    )
            else:
                raise ConfigError(
                    f"{model_id}: min_samples_leaf must be a positive integer or fraction in (0, 0.5]."
                )
        if name == "n_jobs" and (
            not isinstance(value, int) or isinstance(value, bool) or value == 0
        ):
            raise ConfigError(
                f"{model_id}: hyperparameter n_jobs must be an integer other than 0."
            )
        if name in {
            "C",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "ccp_alpha",
        } and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or (name != "ccp_alpha" and value == 0)
        ):
            raise ConfigError(
                f"{model_id}: hyperparameter {name} must be a positive number (or non-negative for ccp_alpha)."
            )
        if name == "random_state" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ConfigError(
                f"{model_id}: hyperparameter random_state must be an integer."
            )
    if parameters.get("class_weight") not in {None, "balanced"}:
        raise ConfigError(
            f"{model_id}: class_weight must be null, 'balanced', or omitted."
        )
    if model_id == "xgboost" and parameters.get("tree_method") not in {
        None,
        "hist",
        "approx",
        "exact",
    }:
        raise ConfigError("xgboost: tree_method must be CPU-compatible.")
