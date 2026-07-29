"""P6C.1B-a contracts remain independent of the nested-CV runner."""

from __future__ import annotations

import copy
import json
import math

import pytest

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.neural import (
    NEURAL_ARTIFACT_SCHEMA_VERSION,
    PROBABILITY_SEMANTICS,
    RESULT_SCOPE,
    validate_neural_early_stopping_split_metadata,
    validate_neural_training_history,
    validate_neural_training_summary,
    validate_training_summary_history,
)


def summary(model_id: str = "mlp_1") -> dict:
    depth = int(model_id[-1])
    return {
        "schema_version": NEURAL_ARTIFACT_SCHEMA_VERSION,
        "experiment_id": "p6c-validation",
        "dataset_id": "german_credit",
        "model_id": model_id,
        "outer_fold_id": "outer_0",
        "inner_fold_id": "inner_0",
        "candidate_id": 2,
        "training_scope": "inner_candidate",
        "hidden_depth": depth,
        "hidden_layers": [64] * depth,
        "parameter_count": 100 + depth,
        "framework": "pytorch",
        "framework_version": "2.12.1",
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 16,
        "max_epochs": 4,
        "epochs_completed": 3,
        "best_epoch": 2,
        "best_validation_loss": 0.3,
        "early_stopping_enabled": True,
        "early_stopping_triggered": False,
        "patience": 2,
        "min_delta": 0.0001,
        "stop_reason": "max_epochs_reached",
        "best_weights_restored": True,
        "duration_seconds": 1.2,
        "model_seed": 42,
        "early_stopping_split_seed": 73,
        "fair_budget_id": "p6b_shared_v1",
        "probability_semantics": PROBABILITY_SEMANTICS,
        "publishable": False,
        "result_scope": RESULT_SCOPE,
        "warnings": [],
    }


def history() -> dict:
    return {
        key: summary()[key]
        for key in (
            "schema_version",
            "experiment_id",
            "dataset_id",
            "model_id",
            "outer_fold_id",
            "inner_fold_id",
            "candidate_id",
            "training_scope",
        )
    } | {
        "epochs": [
            {
                "epoch": 1,
                "train_loss": 0.8,
                "validation_loss": 0.4,
                "learning_rate": 0.001,
                "duration_seconds": 0.2,
                "improved": True,
                "finite": True,
            },
            {
                "epoch": 2,
                "train_loss": 0.6,
                "validation_loss": 0.3,
                "learning_rate": 0.001,
                "duration_seconds": 0.2,
                "improved": True,
                "finite": True,
            },
            {
                "epoch": 3,
                "train_loss": 0.5,
                "validation_loss": 0.35,
                "learning_rate": 0.001,
                "duration_seconds": 0.2,
                "improved": False,
                "finite": True,
            },
        ]
    }


def split_metadata(scope: str = "inner_candidate") -> dict:
    return {
        "schema_version": NEURAL_ARTIFACT_SCHEMA_VERSION,
        "experiment_id": "p6c-validation",
        "dataset_id": "german_credit",
        "model_id": "mlp_1",
        "outer_fold_id": "outer_0",
        "inner_fold_id": "inner_0" if scope == "inner_candidate" else None,
        "candidate_id": 2 if scope == "inner_candidate" else None,
        "split_scope": scope,
        "source_partition": "outer_train" if scope == "final_refit" else "inner_train",
        "strategy": "stratified_holdout",
        "validation_fraction": 0.2,
        "shuffle": True,
        "split_seed": 73,
        "split_hash": "a" * 64,
        "source_row_count": 10,
        "train_row_count": 8,
        "validation_row_count": 2,
        "source_class_counts": {"0": 6, "1": 4},
        "train_class_counts": {"0": 5, "1": 3},
        "validation_class_counts": {"0": 1, "1": 1},
        "overlap_count": 0,
        "union_matches_source": True,
        "publishable": False,
        "result_scope": RESULT_SCOPE,
    }


@pytest.mark.parametrize("model_id", ["mlp_1", "mlp_3", "mlp_5"])
def test_valid_summary_for_each_mlp_depth_and_json_round_trip(model_id: str):
    value = summary(model_id)
    assert validate_neural_training_summary(json.loads(json.dumps(value))) == value


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda p: p.pop("optimizer"), "missing"),
        (lambda p: p.__setitem__("model_id", "mlp_2"), "model_id"),
        (lambda p: p.__setitem__("hidden_depth", 3), "hidden_depth"),
        (lambda p: p.__setitem__("hidden_layers", [64, 64]), "hidden_layers"),
        (lambda p: p.__setitem__("parameter_count", 0), "parameter_count"),
        (lambda p: p.__setitem__("epochs_completed", 5), "epochs_completed"),
        (lambda p: p.__setitem__("best_epoch", 4), "best_epoch"),
        (
            lambda p: p.__setitem__("best_validation_loss", math.nan),
            "best_validation_loss",
        ),
        (lambda p: p.__setitem__("publishable", True), "publishable"),
        (lambda p: p.__setitem__("result_scope", "model_validation"), "result_scope"),
        (
            lambda p: p.__setitem__("probability_semantics", "P(default)"),
            "probability_semantics",
        ),
        (lambda p: p.__setitem__("warnings", [{"state_dict": {}}]), "state_dict"),
    ],
)
def test_summary_rejects_invalid_values(change, field: str):
    value = summary()
    change(value)
    with pytest.raises(ArtifactError, match=field):
        validate_neural_training_summary(value)


def test_history_validates_and_round_trips():
    value = history()
    assert validate_neural_training_history(json.loads(json.dumps(value))) == value


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda p: p["epochs"][0].__setitem__("epoch", 0), "epochs"),
        (lambda p: p["epochs"][1].__setitem__("epoch", 1), "epochs"),
        (lambda p: p["epochs"][2].__setitem__("epoch", 4), "epochs"),
        (lambda p: p["epochs"][0].__setitem__("train_loss", math.inf), "train_loss"),
        (
            lambda p: p["epochs"][0].__setitem__("validation_loss", math.nan),
            "validation_loss",
        ),
        (lambda p: p["epochs"][0].__setitem__("learning_rate", 0), "learning_rate"),
        (
            lambda p: p["epochs"][0].__setitem__("duration_seconds", -1),
            "duration_seconds",
        ),
        (lambda p: p["epochs"][0].__setitem__("finite", False), "finite"),
    ],
)
def test_history_rejects_invalid_epoch_records(change, field: str):
    value = history()
    change(value)
    with pytest.raises(ArtifactError, match=field):
        validate_neural_training_history(value)


def test_summary_history_cross_validation_and_errors():
    valid_summary, valid_history = summary(), history()
    assert validate_training_summary_history(valid_summary, valid_history) is None
    for change, field in [
        (lambda p: p.__setitem__("candidate_id", 9), "candidate_id"),
        (lambda p: p["epochs"].pop(), "count"),
        (
            lambda p: p["epochs"].__setitem__(
                1, {**p["epochs"][1], "validation_loss": 0.31}
            ),
            "best_validation_loss",
        ),
    ]:
        changed_history = copy.deepcopy(valid_history)
        change(changed_history)
        with pytest.raises(ArtifactError, match=field):
            validate_training_summary_history(valid_summary, changed_history)
    missing_best_epoch = copy.deepcopy(valid_summary)
    missing_best_epoch["best_epoch"] = 4
    with pytest.raises(ArtifactError, match="best_epoch"):
        validate_training_summary_history(missing_best_epoch, valid_history)


@pytest.mark.parametrize("scope", ["inner_candidate", "final_refit"])
def test_valid_split_metadata_for_inner_and_final_refit(scope: str):
    value = split_metadata(scope)
    assert (
        validate_neural_early_stopping_split_metadata(json.loads(json.dumps(value)))
        == value
    )


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda p: p.__setitem__("split_hash", "A" * 64), "split_hash"),
        (lambda p: p.__setitem__("train_row_count", 7), "source_row_count"),
        (lambda p: p["train_class_counts"].__setitem__("0", 4), "class_counts"),
        (lambda p: p.__setitem__("overlap_count", 1), "overlap_count"),
        (
            lambda p: p.__setitem__("union_matches_source", False),
            "union_matches_source",
        ),
        (lambda p: p.__setitem__("source_partition", "outer_test"), "source_partition"),
        (lambda p: p.__setitem__("publishable", True), "publishable"),
        (lambda p: p.__setitem__("result_scope", "model_validation"), "result_scope"),
        (
            lambda p: p.__setitem__("source_class_counts", {"state_dict": {}}),
            "state_dict",
        ),
    ],
)
def test_split_metadata_rejects_invalid_values(change, field: str):
    value = split_metadata()
    change(value)
    with pytest.raises(ArtifactError, match=field):
        validate_neural_early_stopping_split_metadata(value)


def test_classical_artifact_modules_remain_importable():
    from creditrep.artifacts import (
        validate_metric_validation_artifact,
        validate_nested_cv_artifact,
    )

    assert callable(validate_metric_validation_artifact)
    assert callable(validate_nested_cv_artifact)
