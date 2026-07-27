"""Single source of truth for stable Phase 5 model identifiers."""

from __future__ import annotations

from typing import Any

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
        unknown = sorted(set(config.hyperparameters) - set(capability.allowed_hyperparameters))
        if unknown:
            raise ConfigError(f"{config.model_id}: unknown hyperparameters: {unknown}.")
        return capability

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return tuple(self._capabilities[key] for key in sorted(self._capabilities))


MODEL_REGISTRY = ModelRegistry()

MODEL_REGISTRY.register(ModelCapability("logistic_regression", "Logistic Regression", "linear", "LogisticRegression", "scikit-learn", True, True, allowed_hyperparameters=("C", "class_weight", "max_iter", "penalty", "random_state", "solver"), implemented=True))
MODEL_REGISTRY.register(ModelCapability("decision_tree", "Decision Tree (implementation pending)", "tree", "DecisionTreeClassifier", "scikit-learn", True, True, allowed_hyperparameters=("criterion", "max_depth", "min_samples_leaf", "min_samples_split", "random_state")))
MODEL_REGISTRY.register(ModelCapability("random_forest", "Random Forest", "ensemble", "RandomForestClassifier", "scikit-learn", True, True, allowed_hyperparameters=("max_depth", "min_samples_leaf", "min_samples_split", "n_estimators", "n_jobs", "random_state")))
MODEL_REGISTRY.register(ModelCapability("xgboost", "XGBoost", "gradient_boosting", "XGBClassifier", "xgboost", True, True, allowed_hyperparameters=("colsample_bytree", "eval_metric", "learning_rate", "max_depth", "n_estimators", "n_jobs", "random_state", "subsample", "tree_method"), implemented=True))
