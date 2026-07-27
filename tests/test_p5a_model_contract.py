from __future__ import annotations

import json

import numpy as np
import pytest

from creditrep.config.exceptions import ConfigError
from creditrep.config.model_config import parse_model_config
from creditrep.models import MODEL_REGISTRY, ModelArtifactMetadata, ModelCapability, ModelRegistry, positive_class_probabilities
from creditrep.models.exceptions import ModelError


class ProbabilityModel:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)


def test_registry_resolves_all_phase_five_ids_and_rejects_unknown_and_duplicate():
    assert {item.model_id for item in MODEL_REGISTRY.capabilities()} == {
        "logistic_regression", "decision_tree", "random_forest", "xgboost"
    }
    assert MODEL_REGISTRY.resolve("logistic_regression").supports_probability is True
    with pytest.raises(ModelError, match="Unknown model ID"):
        MODEL_REGISTRY.resolve("unknown")
    registry = ModelRegistry()
    capability = ModelCapability("logistic_regression", "LR", "linear", "LR", "sklearn", True, True)
    registry.register(capability)
    with pytest.raises(ModelError, match="Duplicate"):
        registry.register(capability)


def test_model_config_validation_and_deterministic_serialization():
    config = parse_model_config({"id": "logistic_regression", "random_seed": 42, "hyperparameters": {"C": 1.0}})
    assert config.to_dict() == parse_model_config({"hyperparameters": {"C": 1.0}, "random_seed": 42, "id": "logistic_regression"}).to_dict()
    with pytest.raises(ConfigError, match="Unknown model ID"):
        parse_model_config({"id": "nope", "random_seed": 1})
    with pytest.raises(ConfigError, match="unknown hyperparameters"):
        parse_model_config({"id": "logistic_regression", "random_seed": 1, "hyperparameters": {"bad": 1}})
    with pytest.raises(ConfigError, match="positive number"):
        parse_model_config({"id": "logistic_regression", "random_seed": 1, "hyperparameters": {"C": 0}})
    with pytest.raises(ConfigError, match="non-empty mapping"):
        parse_model_config({})


def test_positive_class_probability_mapping_and_contract_failures():
    assert positive_class_probabilities(ProbabilityModel([0, 1]), [[0.8, 0.2]], expected_rows=1).tolist() == [0.2]
    assert positive_class_probabilities(ProbabilityModel([1, 0]), [[0.2, 0.8]], expected_rows=1).tolist() == [0.2]
    with pytest.raises(ModelError, match="include class 1"):
        positive_class_probabilities(ProbabilityModel([0, 2]), [[0.4, 0.6]], expected_rows=1)
    with pytest.raises(ModelError, match=r"expected \(n_rows"):
        positive_class_probabilities(ProbabilityModel([0, 1]), [0.1, 0.9], expected_rows=1)
    with pytest.raises(ModelError, match="NaN or Infinity"):
        positive_class_probabilities(ProbabilityModel([0, 1]), [[0.1, np.nan]], expected_rows=1)
    with pytest.raises(ModelError, match=r"\[0, 1\]"):
        positive_class_probabilities(ProbabilityModel([0, 1]), [[1.1, 0.2]], expected_rows=1)
    with pytest.raises(ModelError, match="row count"):
        positive_class_probabilities(ProbabilityModel([0, 1]), [[0.1, 0.9]], expected_rows=2)


def test_model_metadata_is_json_safe_and_non_publishable():
    metadata = ModelArtifactMetadata(
        model_id="logistic_regression", model_family="linear", estimator_name="LogisticRegression",
        library_name="scikit-learn", library_version="1", configured_hyperparameters={"C": 1.0},
        effective_hyperparameters={"C": 1.0}, random_seed=42, expected_classes=(0, 1), observed_classes=(0, 1),
        result_scope="model_validation", publishable=False,
    ).to_dict()
    assert metadata["positive_class"] == 1
    assert metadata["probability_mapping"] == "P(class 1) = P(bad/default)"
    assert metadata["publishable"] is False
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
