from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from creditrep.preprocessing import (
    IterativeVIFSelector,
    LeakageSafePreprocessor,
    PreprocessingError,
    ProtocolAConfig,
    ProtocolAPreprocessingPipeline,
    TrainOnlyStandardScaler,
    WeightOfEvidenceEncoder,
    parse_protocol_a_config,
)


def test_woe_hand_computed_good_over_bad() -> None:
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "a", "b", "b"]})
    y = pd.Series([0, 1, 1, 1])
    encoder = WeightOfEvidenceEncoder(categorical_features=["cat"], passthrough_features=["num"], smoothing=0.5)

    transformed = encoder.fit_transform(X, y)

    expected_a = math.log(((1 + 0.5) / (1 + 0.5 * 2)) / ((1 + 0.5) / (3 + 0.5 * 2)))
    expected_b = math.log(((0 + 0.5) / (1 + 0.5 * 2)) / ((2 + 0.5) / (3 + 0.5 * 2)))
    assert transformed.loc[0, "cat"] == pytest.approx(expected_a)
    assert transformed.loc[2, "cat"] == pytest.approx(expected_b)
    assert encoder.get_metadata()["sign_convention"] == "good_over_bad"


def test_woe_is_finite_for_single_class_categories_and_unknown_is_neutral() -> None:
    X = pd.DataFrame({"cat": ["good_only", "bad_only", "good_only", "bad_only"]})
    y = pd.Series([0, 1, 0, 1])
    encoder = WeightOfEvidenceEncoder(categorical_features=["cat"], smoothing=0.5).fit(X, y)

    transformed, diagnostics = encoder.transform_with_diagnostics(pd.DataFrame({"cat": ["new", "good_only"]}))

    assert np.isfinite(list(encoder.get_metadata()["woe_mapping"]["cat"].values())).all()
    assert transformed.loc[0, "cat"] == pytest.approx(0.0)
    assert diagnostics["unseen_category_counts"] == {"cat": 1}


def test_woe_is_train_only_and_metadata_stable() -> None:
    train = pd.DataFrame({"cat": ["a", "a", "b", "b"]})
    y = pd.Series([0, 0, 1, 1])
    validation = pd.DataFrame({"cat": ["a", "validation_only", "validation_only"]})
    encoder = WeightOfEvidenceEncoder(categorical_features=["cat"], smoothing=0.5).fit(train, y)
    before = encoder.get_metadata()

    first = encoder.transform(validation)
    second = encoder.transform(validation)

    pd.testing.assert_frame_equal(first, second)
    assert encoder.get_metadata() == before
    assert "validation_only" not in encoder.get_metadata()["woe_mapping"]["cat"]


def test_woe_invalid_targets_fail_fast() -> None:
    X = pd.DataFrame({"cat": ["a", "b", "c"]})
    encoder = WeightOfEvidenceEncoder(categorical_features=["cat"])
    with pytest.raises(PreprocessingError, match="exactly classes"):
        encoder.fit(X, pd.Series([0, 1, 2]))
    with pytest.raises(PreprocessingError, match="exactly classes"):
        encoder.fit(X.iloc[:2], pd.Series([0, 0]))
    with pytest.raises(PreprocessingError, match="missing"):
        encoder.fit(X, pd.Series([0, None, 1]))
    with pytest.raises(PreprocessingError, match="length mismatch"):
        encoder.fit(X, pd.Series([0, 1]))


def test_woe_transform_before_fit_and_schema_mismatch_fail_fast() -> None:
    encoder = WeightOfEvidenceEncoder(categorical_features=["cat"])
    with pytest.raises(PreprocessingError, match="before fit"):
        encoder.transform(pd.DataFrame({"cat": ["a"]}))
    encoder.fit(pd.DataFrame({"cat": ["a", "b"]}), pd.Series([0, 1]))
    with pytest.raises(PreprocessingError, match="schema mismatch"):
        encoder.transform(pd.DataFrame({"cat": ["a"], "extra": [1]}))


def test_woe_metadata_serializes() -> None:
    encoder = WeightOfEvidenceEncoder(categorical_features=["cat"]).fit(
        pd.DataFrame({"cat": ["a", "b"]}),
        pd.Series([0, 1]),
    )

    json.dumps(encoder.get_metadata(), allow_nan=False)


def test_vif_removes_duplicate_feature_deterministically() -> None:
    X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [1, 2, 3, 4, 5], "x3": [1, 1, 2, 3, 5]})
    selector = IterativeVIFSelector(threshold=10.0).fit(X)

    metadata = selector.get_metadata()

    assert metadata["removal_history"][0]["removed_feature"] == "x1"
    assert "x1" not in metadata["selected_features"]


def test_vif_keeps_features_below_threshold() -> None:
    X = pd.DataFrame({"x1": [1, -1, 1, -1], "x2": [1, 1, -1, -1]})
    selector = IterativeVIFSelector(threshold=10.0).fit(X)

    assert selector.get_metadata()["removed_features"] == []
    pd.testing.assert_frame_equal(selector.transform(X), X.astype(float))


def test_vif_iterative_removal_and_tie_break() -> None:
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6],
            "x2": [1, 2, 3, 4, 5, 6],
            "x3": [2, 4, 6, 8, 10, 12],
            "x4": [1, 0, 1, 0, 1, 0],
        }
    )
    selector = IterativeVIFSelector(threshold=10.0, minimum_features_to_keep=1).fit(X)
    history = selector.get_metadata()["removal_history"]

    assert len(history) >= 2
    assert history[0]["removed_feature"] == "x1"


def test_vif_constant_feature_and_metadata_immutability() -> None:
    X = pd.DataFrame({"const": [1, 1, 1, 1], "x": [1, 2, 3, 4]})
    selector = IterativeVIFSelector(threshold=10.0).fit(X)
    before = selector.get_metadata()

    transformed = selector.transform(pd.DataFrame({"const": [9, 9], "x": [5, 6]}))

    assert list(transformed.columns) == ["x"]
    assert before == selector.get_metadata()
    assert before["removal_history"][0]["reason"] == "zero_variance"


def test_vif_single_constant_feature_fails_fast() -> None:
    X = pd.DataFrame({"const": [1.0, 1.0, 1.0, 1.0]})

    with pytest.raises(PreprocessingError, match="zero-variance|zero variance|No usable features"):
        IterativeVIFSelector().fit(X)


def test_vif_all_constant_features_fail_fast() -> None:
    X = pd.DataFrame(
        {
            "const_1": [1.0, 1.0, 1.0, 1.0],
            "const_2": [2.0, 2.0, 2.0, 2.0],
        }
    )

    with pytest.raises(PreprocessingError, match="zero-variance|zero variance|No usable features"):
        IterativeVIFSelector().fit(X)


def test_vif_removes_constants_before_vif_and_keeps_usable_feature() -> None:
    X = pd.DataFrame(
        {
            "const_1": [1.0, 1.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "const_2": [2.0, 2.0, 2.0, 2.0],
        }
    )

    selector = IterativeVIFSelector().fit(X)
    metadata = selector.get_metadata()
    zero_variance_removed = [
        item["removed_feature"]
        for item in metadata["removal_history"]
        if item["reason"] == "zero_variance"
    ]

    assert metadata["selected_features"] == ["x"]
    assert zero_variance_removed == ["const_1", "const_2"]
    assert metadata["variance_epsilon"] > 0


def test_vif_single_nonconstant_feature_is_kept() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    selector = IterativeVIFSelector().fit(X)

    assert selector.get_metadata()["selected_features"] == ["x"]


def test_vif_constant_filtering_metadata_stability_and_no_input_mutation() -> None:
    X = pd.DataFrame(
        {
            "const_1": [1.0, 1.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "const_2": [2.0, 2.0, 2.0, 2.0],
        }
    )
    X_before = X.copy(deep=True)
    selector = IterativeVIFSelector().fit(X)
    metadata_before = selector.get_metadata()

    first = selector.transform(X)
    second = selector.transform(X)

    assert metadata_before == selector.get_metadata()
    assert list(first.columns) == ["x"]
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(X, X_before)


def test_vif_minimum_features_to_keep_must_fit_usable_features_after_constant_filtering() -> None:
    X = pd.DataFrame(
        {
            "const": [1.0, 1.0, 1.0, 1.0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "z": [4.0, 3.0, 2.0, 1.0],
        }
    )

    with pytest.raises(PreprocessingError, match="minimum_features_to_keep"):
        IterativeVIFSelector(minimum_features_to_keep=3).fit(X)


def test_vif_train_only_selection_and_schema_validation() -> None:
    train = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [1, 2, 3, 4], "x3": [0, 1, 0, 1]})
    validation = pd.DataFrame({"x1": [1, 0, 1, 0], "x2": [0, 1, 0, 1], "x3": [7, 8, 9, 10]})
    selector = IterativeVIFSelector(threshold=10.0).fit(train)
    selected = selector.get_metadata()["selected_features"]

    selector.transform(validation)

    assert selector.get_metadata()["selected_features"] == selected
    with pytest.raises(PreprocessingError, match="schema mismatch"):
        selector.transform(validation.drop(columns=["x2"]))


def test_vif_single_feature_and_more_features_than_rows() -> None:
    single = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    single_selector = IterativeVIFSelector().fit(single)
    assert single_selector.get_metadata()["selected_features"] == ["x"]

    wide = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0], "c": [3.0, 6.0]})
    wide_selector = IterativeVIFSelector(threshold=10.0, minimum_features_to_keep=1).fit(wide)
    assert len(wide_selector.get_metadata()["selected_features"]) >= 1


def test_vif_rejects_empty_non_numeric_and_non_finite() -> None:
    with pytest.raises(PreprocessingError, match="empty"):
        IterativeVIFSelector().fit(pd.DataFrame())
    with pytest.raises(PreprocessingError, match="numeric"):
        IterativeVIFSelector().fit(pd.DataFrame({"x": ["a", "b"]}))
    with pytest.raises(PreprocessingError, match="NaN"):
        IterativeVIFSelector().fit(pd.DataFrame({"x": [1.0, np.nan]}))


def test_scaler_uses_training_statistics_and_metadata_stable() -> None:
    train = pd.DataFrame({"x": [1.0, 3.0]})
    validation = pd.DataFrame({"x": [None]})
    imputer = LeakageSafePreprocessor(numeric_features=["x"], categorical_features=[]).fit(train)
    train_imputed = imputer.transform(train)
    validation_imputed = imputer.transform(validation)
    scaler = TrainOnlyStandardScaler(enabled=True).fit(train_imputed)
    before = scaler.get_metadata()

    transformed = scaler.transform(validation_imputed)

    assert transformed.loc[0, "x"] == pytest.approx(0.0)
    assert scaler.get_metadata() == before


def test_scaler_disabled_passthrough_and_serializes() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0], "z": [5.0, 5.0]})
    scaler = TrainOnlyStandardScaler(enabled=False).fit(X)

    pd.testing.assert_frame_equal(scaler.transform(X), X)
    metadata = scaler.get_metadata()
    assert metadata["enabled"] is False
    assert metadata["zero_variance_features"] == ["z"]
    json.dumps(metadata, allow_nan=False)


def test_scaler_edge_cases_and_schema_validation() -> None:
    scaler = TrainOnlyStandardScaler(enabled=True).fit(pd.DataFrame({"x": [1.0]}))
    assert scaler.transform(pd.DataFrame({"x": [2.0]})).loc[0, "x"] == pytest.approx(1.0)
    with pytest.raises(PreprocessingError, match="schema mismatch"):
        scaler.transform(pd.DataFrame({"x": [1.0], "extra": [1.0]}))
    with pytest.raises(PreprocessingError, match="before fit"):
        TrainOnlyStandardScaler().transform(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(PreprocessingError, match="NaN"):
        scaler.transform(pd.DataFrame({"x": [np.nan]}))


def test_protocol_a_config_p3b_validation() -> None:
    payload = {
        "protocol": {
            "name": "protocol_a",
            "version": "p3b-v1",
            "numeric_imputation": {"strategy": "mean"},
            "categorical_imputation": {"strategy": "most_frequent"},
            "unseen_category": {"strategy": "reserved_token", "token": "__UNKNOWN__"},
            "woe": {"enabled": True, "scope": "categorical", "smoothing": 0.5, "unknown_value": 0.0},
            "vif": {"enabled": True, "threshold": 10.0, "minimum_features_to_keep": 1},
            "scaling": {"enabled": False, "strategy": "standard"},
        }
    }

    config = parse_protocol_a_config(payload)

    assert config.woe_enabled is True
    assert config.vif_enabled is True
    payload["protocol"]["woe"]["smoothing"] = 0
    with pytest.raises(PreprocessingError, match="smoothing"):
        parse_protocol_a_config(payload)


def test_end_to_end_p3b_pipeline_numeric_finite_deterministic_and_metadata_stable() -> None:
    train = pd.DataFrame(
        {
            "num": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "dup": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "cat": ["a", "a", "b", "b", "c", "c"],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1])
    validation = pd.DataFrame({"num": [None, 10.0], "dup": [None, 10.0], "cat": ["new", "a"]})
    validation_before = validation.copy(deep=True)
    config = ProtocolAConfig(woe_enabled=True, vif_enabled=True, scaling_enabled=True)
    pipeline = ProtocolAPreprocessingPipeline(
        numeric_features=["num", "dup"],
        categorical_features=["cat"],
        config=config,
    ).fit(train, y)
    before = pipeline.get_metadata()

    transformed, diagnostics = pipeline.transform_with_diagnostics(validation)
    second = pipeline.transform(validation)
    clone = ProtocolAPreprocessingPipeline(
        numeric_features=["num", "dup"],
        categorical_features=["cat"],
        config=config,
    ).fit(train, y)

    assert pipeline.get_metadata() == before
    assert diagnostics["imputation"]["unseen_category_counts"] == {"cat": 1}
    assert np.isfinite(transformed.to_numpy()).all()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in transformed.dtypes)
    pd.testing.assert_frame_equal(transformed, second)
    pd.testing.assert_frame_equal(transformed, clone.transform(validation))
    pd.testing.assert_frame_equal(validation, validation_before)
    json.dumps(before, allow_nan=False)
