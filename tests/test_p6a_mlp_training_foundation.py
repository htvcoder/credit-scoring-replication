"""CPU-only contracts for the P6A neural training foundation."""

import json

import numpy as np
import pytest

from creditrep.models.neural import MLPConfig, parse_mlp_config, train_mlp
from creditrep.models.neural.exceptions import DevicePolicyError, MLPConfigError, NeuralInputError
from creditrep.models.neural.network import ConfigurableMLP, architecture_metadata
from creditrep.models.neural.runtime import configure_runtime, require_torch, resolve_device
from creditrep.models.neural.training import EarlyStoppingState


def config(**changes):
    values = {"hidden_layers": (4,), "random_seed": 9, "device_policy": "cpu", "max_epochs": 12, "batch_size": 4}
    values.update(changes)
    return MLPConfig(**values)


@pytest.mark.parametrize("changes", [{"hidden_layers": ()}, {"hidden_layers": (0,)}, {"dropout": 1.0}, {"learning_rate": 0}, {"weight_decay": -1}, {"batch_size": 0}, {"max_epochs": 0}, {"device_policy": "gpu"}, {"activation": "tanh"}, {"optimizer": "sgd"}, {"dataloader_workers": -1}])
def test_config_rejects_invalid_values(changes):
    with pytest.raises(MLPConfigError):
        config(**changes)


def test_config_is_json_safe_and_unknown_keys_fail():
    assert json.loads(json.dumps(config().to_dict()))["hidden_layers"] == [4]
    with pytest.raises(MLPConfigError, match="unsupported keys"):
        parse_mlp_config({"hidden_layers": [4], "random_seed": 1, "unknown": True})


@pytest.mark.parametrize("layers", [(4,), (4, 4, 4), (4, 4, 4, 4, 4)])
def test_builder_returns_logits_and_json_architecture(layers):
    torch = require_torch()
    model = ConfigurableMLP(3, config(hidden_layers=layers, dropout=0.1, batch_normalization=True))
    output = model(torch.zeros((2, 3)))
    metadata = architecture_metadata(model, input_dimension=3, config=config(hidden_layers=layers))
    assert output.shape == (2, 1)
    assert not any(isinstance(item, torch.nn.Sigmoid) for item in model.modules())
    assert metadata["trainable_parameter_count"] > 0
    json.dumps(metadata)


def test_runtime_cpu_and_cuda_policy_are_explicit():
    assert str(resolve_device("cpu")) == "cpu"
    assert str(resolve_device("auto")) in {"cpu", "cuda"}
    if not require_torch().cuda.is_available():
        with pytest.raises(DevicePolicyError, match="explicitly requested"):
            resolve_device("cuda")


def test_same_seed_produces_same_cpu_initial_weights():
    configure_runtime(seed=17, device_policy="cpu", deterministic=True)
    first = [value.detach().clone() for value in ConfigurableMLP(2, config()).parameters()]
    configure_runtime(seed=17, device_policy="cpu", deterministic=True)
    second = [value.detach().clone() for value in ConfigurableMLP(2, config()).parameters()]
    assert all(require_torch().equal(a, b) for a, b in zip(first, second))


def test_early_stopping_min_delta_and_non_finite_contract():
    state = EarlyStoppingState(patience=2, min_delta=0.1)
    assert state.update(1.0, 1)
    assert not state.update(0.95, 2)
    assert not state.triggered
    assert not state.update(0.95, 3)
    assert state.triggered and state.stop_reason == "early_stopping_patience_exhausted"
    non_finite = EarlyStoppingState(patience=2, min_delta=0.0)
    assert not non_finite.update(float("nan"), 1)
    assert non_finite.stop_reason == "non_finite_validation_loss"


def test_tiny_cpu_training_prediction_and_metadata():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., .2], [.2, 0.], [.8, 1.], [1., .8]])
    y = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    result = train_mlp(X, y, config(max_epochs=16), X_validation=X, y_validation=y)
    probabilities = result.predict_probabilities(X)
    assert probabilities.shape == (len(X),)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert result.summary.loss_function == "BCEWithLogitsLoss"
    assert result.summary.publishable is False
    assert result.summary.runtime.requested_device == "cpu"
    assert result.summary.best_weights_restored
    assert len(result.history.epochs) > 0
    json.dumps(result.history.to_dict())
    json.dumps(result.summary.to_dict())


@pytest.mark.parametrize("X,y", [(np.array([1, 2]), np.array([0, 1])), (np.array([[1., np.nan]]), np.array([0])), (np.array([[1.], [2.]]), np.array([0, 2]))])
def test_invalid_tiny_training_inputs_fail_fast(X, y):
    with pytest.raises(NeuralInputError):
        train_mlp(X, y, config(), X_validation=np.array([[0.]]), y_validation=np.array([0]))


def test_validation_feature_mismatch_and_missing_validation_fail_fast():
    with pytest.raises(NeuralInputError, match="feature dimension mismatch"):
        train_mlp(np.ones((3, 2)), np.array([0, 1, 0]), config(), X_validation=np.ones((2, 3)), y_validation=np.array([0, 1]))
    with pytest.raises(NeuralInputError, match="requires validation"):
        train_mlp(np.ones((3, 2)), np.array([0, 1, 0]), config())
