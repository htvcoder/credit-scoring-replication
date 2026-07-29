import json
import numpy as np
import pytest
from creditrep.models import MODEL_REGISTRY, create_model
from creditrep.models.neural.exceptions import MLPConfigError, NeuralInputError
from creditrep.models.neural.specifications import MLP_SPECS


def test_specs_ids_depths_and_fair_budget_json_safe():
    assert set(MLP_SPECS) == {"mlp_1", "mlp_3", "mlp_5"}
    assert [len(x.hidden_layers) for x in MLP_SPECS.values()] == [1, 3, 5]
    assert {x.to_dict()["training_budget_id"] for x in MLP_SPECS.values()} == {
        "p6b_shared_v1"
    }
    json.dumps([x.to_dict() for x in MLP_SPECS.values()])


def test_depth_mismatch_fails():
    with pytest.raises(MLPConfigError):
        MLP_SPECS["mlp_1"].config(hidden_layers=(64, 64))


@pytest.mark.parametrize("model_id", ["mlp_1", "mlp_3", "mlp_5"])
def test_factory_toy_cpu_fit_probability_metadata(model_id):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    model = create_model(
        model_id,
        {"device_policy": "cpu", "max_epochs": 3, "batch_size": 2},
        random_seed=7,
    )
    model.fit(X, y, X_validation=X, y_validation=y)
    p = model.predict_proba(X)
    assert (
        p.shape == (4, 2)
        and np.isfinite(p).all()
        and ((p >= 0) & (p <= 1)).all()
        and model.get_model_metadata()["hidden_depth"] == int(model_id[-1])
    )


def test_predict_before_fit_and_feature_mismatch_fail():
    model = create_model("mlp_1", {"device_policy": "cpu", "max_epochs": 1})
    with pytest.raises(NeuralInputError):
        model.predict_proba(np.ones((1, 2)))
    model.fit(
        np.ones((2, 2)),
        np.array([0, 1]),
        X_validation=np.ones((2, 2)),
        y_validation=np.array([0, 1]),
    )
    with pytest.raises(NeuralInputError):
        model.predict_proba(np.ones((1, 3)))


def test_registry_keeps_classical_ids():
    assert {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "mlp_1",
        "mlp_3",
        "mlp_5",
    }.issubset({x.model_id for x in MODEL_REGISTRY.capabilities()})
