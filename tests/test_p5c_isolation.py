"""Nested-CV leakage and fresh-refit acceptance tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import creditrep.experiments.model_validation as runner
from creditrep.config.model_validation import parse_model_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition


def _inputs(target=None):
    target = pd.Series([0, 1] * 12) if target is None else target
    dataset = LoadedDataset("TOY", pd.DataFrame({"x": range(24)}), target, {"numeric_columns": ["x"], "categorical_columns": [], "source_file": "fixture.csv", "row_count": 24, "feature_count": 1}, Path("fixture.csv"))
    config = parse_model_validation_config({"experiment": {"name": "isolation", "publishable": False, "result_scope": "model_validation"}, "dataset": {"id": "TOY"}, "output": {"root_dir": "ignored"}, "models": {"logistic_regression": [{"C": 0.1, "max_iter": 30, "solver": "liblinear"}, {"C": 1.0, "max_iter": 30, "solver": "liblinear"}]}, "metrics": [{"id": "roc_auc"}], "optimization_metric": "roc_auc", "random_seed": 17})
    nested = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats=1, inner_n_splits=2, random_seed=17)
    return dataset, config, nested


def test_inner_folds_are_strictly_contained_in_outer_training_partition():
    _, _, nested = _inputs()
    for outer in nested.outer_folds:
        train, test = set(outer.train_indices), set(outer.test_indices)
        for inner in outer.inner_folds:
            assert set(inner.train_indices) <= train and set(inner.validation_indices) <= train
            assert not (set(inner.train_indices) & test) and not (set(inner.validation_indices) & test)
            assert not (set(inner.train_indices) & set(inner.validation_indices))


def test_candidate_selection_is_independent_of_outer_test_labels():
    dataset, config, nested = _inputs(); outer = nested.outer_folds[0]
    baseline = runner._run_fold(config=config, dataset=dataset, outer=outer, model_id="logistic_regression", candidates=config.model_candidates["logistic_regression"], protocol_config=ProtocolAConfig())
    changed = dataset.target.copy(); changed.iloc[list(outer.test_indices)] = 1 - changed.iloc[list(outer.test_indices)]
    changed_dataset = LoadedDataset(dataset.dataset_id, dataset.features, changed, dataset.metadata, Path("fixture.csv"))
    altered = runner._run_fold(config=config, dataset=changed_dataset, outer=outer, model_id="logistic_regression", candidates=config.model_candidates["logistic_regression"], protocol_config=ProtocolAConfig())
    assert baseline["selected_candidate"] == altered["selected_candidate"]


def test_preprocessing_is_fit_only_on_inner_then_outer_train_rows(monkeypatch):
    dataset, config, nested = _inputs(); outer = nested.outer_folds[0]; calls = []
    original = runner._fit_preprocessing
    def spy(dataset_arg, *, train_indices, transform_indices, protocol_config):
        calls.append((set(train_indices), set(transform_indices)))
        return original(dataset_arg, train_indices=train_indices, transform_indices=transform_indices, protocol_config=protocol_config)
    monkeypatch.setattr(runner, "_fit_preprocessing", spy)
    runner._run_fold(config=config, dataset=dataset, outer=outer, model_id="logistic_regression", candidates=config.model_candidates["logistic_regression"], protocol_config=ProtocolAConfig())
    assert all(train <= set(outer.train_indices) and not train & set(outer.test_indices) for train, _ in calls)
    assert calls[-1] == (set(outer.train_indices), set(outer.test_indices))


def test_outer_refit_uses_fresh_estimator_and_predicts_after_fit(monkeypatch):
    dataset, config, nested = _inputs(); outer = nested.outer_folds[0]; events = []; original = runner.create_model
    class Spy:
        def __init__(self, identifier, wrapped): self.identifier, self.wrapped = identifier, wrapped
        def fit(self, X, y): events.append(("fit", self.identifier, len(X))); self.wrapped.fit(X, y); return self
        def predict_proba(self, X): events.append(("predict", self.identifier, len(X))); return self.wrapped.predict_proba(X)
        def __getattr__(self, name): return getattr(self.wrapped, name)
    def factory(*args, **kwargs):
        return Spy(len([event for event in events if event[0] == "create"]), original(*args, **kwargs))
    # Track creates independently so final refit must be a new instance.
    counter = {"value": 0}
    def counted_factory(*args, **kwargs):
        counter["value"] += 1; return Spy(counter["value"], original(*args, **kwargs))
    monkeypatch.setattr(runner, "create_model", counted_factory)
    runner._run_fold(config=config, dataset=dataset, outer=outer, model_id="logistic_regression", candidates=config.model_candidates["logistic_regression"], protocol_config=ProtocolAConfig())
    final_id = counter["value"]
    assert ("fit", final_id, len(outer.train_indices)) in events
    assert events.index(("fit", final_id, len(outer.train_indices))) < events.index(("predict", final_id, len(outer.test_indices)))
