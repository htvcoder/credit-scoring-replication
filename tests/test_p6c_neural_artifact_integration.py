"""P6C.1B-b neural fold-set and publication regression tests."""

from __future__ import annotations

import copy
import json

import pytest
import pandas as pd

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.neural import (
    NEURAL_ARTIFACT_SCHEMA_VERSION,
    RESULT_SCOPE,
    validate_neural_fold_artifact_set,
)
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.experiments.model_validation import run_folded_model_validation
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition
from tests.test_p6c_neural_artifact_contracts import history, split_metadata, summary


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fold(tmp_path):
    _write(tmp_path / "model_metadata.json", {"model_id": "mlp_1"})
    candidate_summary = summary()
    candidate_history = history()
    candidate_split = split_metadata()
    final_summary = copy.deepcopy(candidate_summary)
    final_history = copy.deepcopy(candidate_history)
    final_split = copy.deepcopy(candidate_split)
    for value in (final_summary, final_history):
        value["inner_fold_id"] = "final_refit"
        value["training_scope"] = "final_refit"
    final_split["inner_fold_id"] = "final_refit"
    final_split["split_scope"] = "final_refit"
    paths = {
        "candidate training_summary": "neural/candidates/candidate-2/inner-inner_0/training_summary.json",
        "candidate training_history": "neural/candidates/candidate-2/inner-inner_0/training_history.json",
        "candidate early_stopping_split": "neural/candidates/candidate-2/inner-inner_0/early_stopping_split.json",
        "final-refit training_summary": "neural/final_refit/training_summary.json",
        "final-refit training_history": "neural/final_refit/training_history.json",
        "final-refit early_stopping_split": "neural/final_refit/early_stopping_split.json",
    }
    _write(tmp_path / paths["candidate training_summary"], candidate_summary)
    _write(tmp_path / paths["candidate training_history"], candidate_history)
    _write(tmp_path / paths["candidate early_stopping_split"], candidate_split)
    _write(tmp_path / paths["final-refit training_summary"], final_summary)
    _write(tmp_path / paths["final-refit training_history"], final_history)
    _write(tmp_path / paths["final-refit early_stopping_split"], final_split)
    _write(
        tmp_path / "neural/neural_manifest.json",
        {
            "schema_version": NEURAL_ARTIFACT_SCHEMA_VERSION,
            "experiment_id": "p6c-validation",
            "dataset_id": "german_credit",
            "model_id": "mlp_1",
            "outer_fold_id": "outer_0",
            "publishable": False,
            "result_scope": RESULT_SCOPE,
            "fair_budget_id": "p6b_shared_v1",
            "selected_candidate_id": 2,
            "candidates": [
                {
                    "candidate_id": 2,
                    "runs": [
                        {
                            "training_summary": paths["candidate training_summary"],
                            "training_history": paths["candidate training_history"],
                            "early_stopping_split": paths[
                                "candidate early_stopping_split"
                            ],
                        }
                    ],
                }
            ],
            "final_refit": {
                "selected_candidate_id": 2,
                "training_summary": paths["final-refit training_summary"],
                "training_history": paths["final-refit training_history"],
                "early_stopping_split": paths["final-refit early_stopping_split"],
            },
        },
    )
    return tmp_path


def test_complete_neural_fold_artifact_set_is_valid(tmp_path):
    validate_neural_fold_artifact_set(_fold(tmp_path))


@pytest.mark.parametrize(
    "relative",
    [
        "neural/candidates/candidate-2/inner-inner_0/training_summary.json",
        "neural/candidates/candidate-2/inner-inner_0/training_history.json",
        "neural/candidates/candidate-2/inner-inner_0/early_stopping_split.json",
        "neural/final_refit/training_summary.json",
    ],
)
def test_missing_neural_evidence_fails_before_publication(tmp_path, relative):
    root = _fold(tmp_path)
    (root / relative).unlink()
    with pytest.raises(ArtifactError, match="missing"):
        validate_neural_fold_artifact_set(root)


def test_classical_fold_needs_no_neural_directory(tmp_path):
    _write(tmp_path / "model_metadata.json", {"model_id": "logistic_regression"})
    validate_neural_fold_artifact_set(tmp_path)


def test_tiny_cpu_mlp_fold_publishes_complete_neural_evidence(tmp_path):
    dataset = LoadedDataset(
        "TOY",
        pd.DataFrame({"x": list(range(24)), "z": [item % 3 for item in range(24)]}),
        pd.Series([0, 1] * 12),
        {
            "numeric_columns": ["x", "z"],
            "categorical_columns": [],
            "source_file": "fixture.csv",
            "row_count": 24,
            "feature_count": 2,
        },
        tmp_path / "fixture.csv",
    )
    config = parse_model_validation_config(
        {
            "experiment": {
                "name": "neural_fixture",
                "publishable": False,
                "result_scope": "model_validation",
            },
            "dataset": {"id": "TOY"},
            "output": {"root_dir": "ignored"},
            "models": {
                "mlp_1": [{"max_epochs": 1, "batch_size": 4, "device_policy": "cpu"}]
            },
            "metrics": [{"id": "roc_auc"}],
            "optimization_metric": "roc_auc",
            "random_seed": 17,
        }
    )
    definition = create_nested_cv_definition(
        dataset,
        dataset_checksum="fixture",
        outer_n_repeats=1,
        inner_n_splits=2,
        random_seed=17,
    )
    root, result = run_folded_model_validation(
        config=config,
        dataset=dataset,
        nested_cv=definition,
        protocol_config=ProtocolAConfig(),
        output_root=tmp_path,
        dataset_checksum="fixture",
    )
    assert result["completed_fold_count"] == 2
    for fold in (root / "folds").iterdir():
        validate_neural_fold_artifact_set(fold)
        assert not list(fold.rglob("*.pt"))
