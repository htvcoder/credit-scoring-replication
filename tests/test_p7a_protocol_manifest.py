from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
import pandas as pd

from creditrep.config.exceptions import ConfigError
from creditrep.models.registry import validate_hyperparameters
from creditrep.protocols.p7a import ProtocolManifestError, effective_min_samples_leaf, load_manifest, manifest_hash, validate_manifest, verify_manifest_lock
from creditrep.datasets.models import LoadedDataset
from creditrep.splitting.nested import create_nested_cv_definition


MANIFEST = Path("configs/protocols/p7a/p7a_candidate_manifest.yaml")


def test_candidate_manifest_is_locked_and_uses_paper_repeat_counts():
    payload = load_manifest(MANIFEST)
    assert payload["cross_validation"]["outer_repeats"] == {"AC": 10, "GC": 10, "TH02": 10, "HMEQ": 5, "TC": 5, "GMC": 5}
    assert payload["reference_search_spaces"]["xgboost"]["declared_configurations"] == 108


def test_mutating_scientific_field_breaks_lock():
    payload = load_manifest(MANIFEST)
    payload["cross_validation"]["seed"] = 43
    with pytest.raises(ProtocolManifestError, match="manifest_sha256"):
        verify_manifest_lock(payload)


def test_unknown_model_and_invalid_override_are_rejected():
    payload = load_manifest(MANIFEST)
    broken = deepcopy(payload)
    broken["models"][0]["id"] = "unknown"
    with pytest.raises(ProtocolManifestError, match="model ID"):
        validate_manifest(broken)
    broken = deepcopy(payload)
    broken["runtime_overrides"]["allowed"].append("seed")
    with pytest.raises(ProtocolManifestError, match="override"):
        validate_manifest(broken)


def test_canonical_hash_is_deterministic():
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest_hash(payload) == manifest_hash(deepcopy(payload))


def test_search_space_combination_count_is_checked():
    payload = load_manifest(MANIFEST)
    payload["reference_search_spaces"]["xgboost"]["parameters"]["subsample"] = [0.5]
    with pytest.raises(ProtocolManifestError, match="tổ hợp grid"):
        validate_manifest(payload)


def test_nested_cv_accepts_dataset_specific_repeat_mapping():
    dataset = LoadedDataset("ac", pd.DataFrame({"x": range(20)}), pd.Series([0, 1] * 10), {}, Path("fixture.csv"))
    definition = create_nested_cv_definition(dataset, dataset_checksum="fixture", outer_n_repeats={"AC": 10}, inner_n_splits=2)
    assert definition.outer_n_repeats == 10


def test_min_samples_leaf_accepts_integer_and_relative_fraction():
    validate_hyperparameters("decision_tree", {"min_samples_leaf": 3})
    validate_hyperparameters("decision_tree", {"min_samples_leaf": 0.005})


@pytest.mark.parametrize("value", [True, 0, -1, 0.0, -0.1, 0.50001, float("nan"), float("inf")])
def test_min_samples_leaf_rejects_invalid_values(value):
    with pytest.raises(ConfigError):
        validate_hyperparameters("decision_tree", {"min_samples_leaf": value})


def test_cart_grid_and_pilot_budget_are_locked():
    payload = load_manifest(MANIFEST)
    cart = payload["candidate_search_space"]["cart_a"]
    candidates = [(depth, leaf) for depth in cart["max_depth"] for leaf in cart["min_samples_leaf"]]
    assert len(candidates) == len(set(candidates)) == 12
    pilot = payload["pilot_budget"]
    pilot_values = {(item["max_depth"], item["min_samples_leaf"]) for item in pilot["candidates"]}
    assert len(pilot_values) == 4
    assert pilot_values <= set(candidates)
    assert pilot["total_inner_fits"] == 3 * 4 * 5 == 60
    assert cart["full_theoretical_workload"]["inner_candidate_evaluation_fits"] == 450 * 12 == 5400
    assert payload["final_scientific_search_space"]["status"] == "created_only_after_p7b_closeout_before_p7c"


def test_effective_minimum_leaf_count_uses_ceiling():
    assert effective_min_samples_leaf(0.005, 276) == 2
    assert effective_min_samples_leaf(0.01, 400) == 4
    assert effective_min_samples_leaf(3, 276) == 3


def test_manifest_round_trip_keeps_canonical_hash():
    payload = load_manifest(MANIFEST)
    reparsed = yaml.safe_load(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
    assert manifest_hash(payload) == manifest_hash(reparsed)
