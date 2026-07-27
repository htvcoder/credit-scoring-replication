from __future__ import annotations

import json
import time

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from creditrep.config.exceptions import ConfigError
from creditrep.config.model_config import parse_model_config
from creditrep.models import MODEL_REGISTRY, build_model_metadata, create_model, positive_class_probabilities
from creditrep.models.exceptions import ModelError


@pytest.fixture
def fixture_data():
    return make_classification(n_samples=40, n_features=5, n_informative=3, random_state=7)


@pytest.mark.parametrize(
    ("model_id", "estimator_type"),
    [("logistic_regression", LogisticRegression), ("decision_tree", DecisionTreeClassifier), ("random_forest", RandomForestClassifier)],
)
def test_factory_builds_classical_estimators_and_probability_contract(fixture_data, model_id, estimator_type):
    X, y = fixture_data
    model = create_model(model_id, {}, random_seed=42)
    assert isinstance(model, estimator_type)
    model.fit(X, y)
    score = positive_class_probabilities(model, model.predict_proba(X), expected_rows=len(y))
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()


def test_xgboost_factory_is_cpu_safe_and_all_capabilities_are_implemented():
    model = create_model("xgboost", {}, random_seed=42)
    assert model.get_params()["tree_method"] == "hist"
    assert all(capability.implemented for capability in MODEL_REGISTRY.capabilities())


def test_factory_and_config_reject_invalid_parameters_and_seed_conflicts():
    with pytest.raises(ConfigError, match="unknown hyperparameters"):
        parse_model_config({"id": "decision_tree", "random_seed": 1, "hyperparameters": {"bad": 1}})
    with pytest.raises(ConfigError, match="n_jobs"):
        parse_model_config({"id": "random_forest", "random_seed": 1, "hyperparameters": {"n_jobs": 0}})
    with pytest.raises(ConfigError, match="conflicts"):
        parse_model_config({"id": "random_forest", "random_seed": 1, "hyperparameters": {"random_state": 2}})
    with pytest.raises(ModelError, match="conflicts"):
        create_model("decision_tree", {"random_state": 2}, random_seed=1)


def test_cart_metadata_is_propagated_from_fitted_estimator(fixture_data):
    X, y = fixture_data
    model = create_model("decision_tree", {}, random_seed=42).fit(X, y)
    metadata = build_model_metadata(model, model_id="decision_tree", configured_hyperparameters={}, random_seed=42, fit_duration_seconds=0.01, prediction_duration_seconds=0.01)
    assert metadata["algorithm"] == "cart"
    assert metadata["implementation"] == "sklearn.tree.DecisionTreeClassifier"
    assert metadata["replication_role"] == "approximation"
    assert metadata["deviation_from_paper"] == "c45_to_cart"
    assert metadata["publishable"] is False and metadata["positive_class"] == 1
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata


@pytest.mark.parametrize("model_id", ["logistic_regression", "decision_tree", "random_forest", "xgboost"])
def test_reduced_profile_defaults_are_deterministic_and_non_balanced(model_id):
    capability = MODEL_REGISTRY.resolve(model_id)
    first = create_model(model_id, {}, random_seed=13).get_params(deep=False)
    second = create_model(model_id, {}, random_seed=13).get_params(deep=False)
    assert first == second
    assert first["random_state"] == 13
    if "class_weight" in first:
        assert first["class_weight"] is None
    assert capability.default_hyperparameters
