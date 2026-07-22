from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditrep.datasets.loader import load_dataset
from creditrep.datasets.registry import load_registry
from creditrep.preprocessing import (
    LeakageSafePreprocessor,
    PreprocessingError,
    features_from_dataset_metadata,
    load_protocol_a_config,
    parse_protocol_a_config,
)


def protocol_payload() -> dict:
    return {
        "protocol": {
            "name": "protocol_a",
            "numeric_imputation": {"strategy": "mean"},
            "categorical_imputation": {"strategy": "most_frequent"},
            "unseen_category": {"strategy": "reserved_token", "token": "__UNKNOWN__"},
        }
    }


def make_preprocessor() -> LeakageSafePreprocessor:
    return LeakageSafePreprocessor(numeric_features=["num"], categorical_features=["cat"])


def test_train_only_numeric_imputation_uses_training_mean() -> None:
    train = pd.DataFrame({"num": [1.0, 3.0, None], "cat": ["a", "a", "b"]})
    test = pd.DataFrame({"num": [None, 999.0], "cat": ["a", "b"]})
    preprocessor = make_preprocessor().fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.loc[0, "num"] == pytest.approx(2.0)
    assert preprocessor.get_metadata()["numeric_imputation_values"] == {"num": 2.0}


def test_train_only_categorical_imputation_uses_training_mode() -> None:
    train = pd.DataFrame({"num": [1.0, 3.0, 5.0], "cat": ["train_mode", "train_mode", "x"]})
    test = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": [None, "test_mode", "test_mode", "test_mode"]})
    preprocessor = make_preprocessor().fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.loc[0, "cat"] == "train_mode"
    assert preprocessor.get_metadata()["categorical_imputation_values"] == {"cat": "train_mode"}


def test_unseen_category_maps_to_unknown_without_learning_new_vocabulary() -> None:
    train = pd.DataFrame({"num": [1.0, 2.0], "cat": ["known", "other"]})
    test = pd.DataFrame({"num": [3.0], "cat": ["test_only"]})
    preprocessor = make_preprocessor().fit(train)
    vocabulary_before = preprocessor.get_metadata()["category_vocabulary"]["cat"]

    transformed, diagnostics = preprocessor.transform_with_diagnostics(test)
    metadata = preprocessor.get_metadata()

    assert transformed.loc[0, "cat"] == "__UNKNOWN__"
    assert metadata["category_vocabulary"]["cat"] == vocabulary_before
    assert diagnostics["unseen_category_counts"] == {"cat": 1}
    assert "unseen_category_counts" not in metadata


def test_mode_tie_break_uses_type_name_and_repr() -> None:
    train = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["b", "a", "b", "a"]})
    preprocessor = make_preprocessor().fit(train)

    assert preprocessor.get_metadata()["categorical_imputation_values"]["cat"] == "a"


def test_mode_tie_break_distinguishes_values_with_same_string_representation() -> None:
    train = pd.DataFrame({"num": [1.0, 2.0], "cat": [1, "1"]})
    preprocessor = make_preprocessor().fit(train)

    assert preprocessor.get_metadata()["categorical_imputation_values"]["cat"] == 1


def test_transform_before_fit_fails_fast() -> None:
    with pytest.raises(PreprocessingError, match="before fit"):
        make_preprocessor().transform(pd.DataFrame({"num": [1.0], "cat": ["a"]}))


def test_schema_mismatch_missing_extra_and_order_fail_fast() -> None:
    preprocessor = make_preprocessor().fit(pd.DataFrame({"num": [1.0], "cat": ["a"]}))
    with pytest.raises(PreprocessingError, match="missing"):
        preprocessor.transform(pd.DataFrame({"num": [1.0]}))
    with pytest.raises(PreprocessingError, match="extra"):
        preprocessor.transform(pd.DataFrame({"num": [1.0], "cat": ["a"], "extra": [1]}))
    with pytest.raises(PreprocessingError, match="column order"):
        preprocessor.transform(pd.DataFrame({"cat": ["a"], "num": [1.0]}))


def test_metadata_dataframe_mismatch_fails_fast() -> None:
    metadata = {
        "target_column": "BAD",
        "removed_columns": ["ID"],
        "numeric_columns": ["num"],
        "categorical_columns": ["cat"],
    }
    with pytest.raises(PreprocessingError, match="not declared"):
        features_from_dataset_metadata(["num", "cat", "extra"], metadata)
    with pytest.raises(PreprocessingError, match="Target column"):
        features_from_dataset_metadata(["num", "cat", "BAD"], metadata)
    with pytest.raises(PreprocessingError, match="Removed"):
        features_from_dataset_metadata(["num", "cat", "ID"], metadata)


def test_all_missing_columns_fail_fast() -> None:
    with pytest.raises(PreprocessingError, match="Numeric feature 'num'"):
        make_preprocessor().fit(pd.DataFrame({"num": [None, None], "cat": ["a", "b"]}))
    with pytest.raises(PreprocessingError, match="Categorical feature 'cat'"):
        make_preprocessor().fit(pd.DataFrame({"num": [1.0, 2.0], "cat": [None, None]}))


def test_no_in_place_mutation_for_fit_and_transform() -> None:
    train = pd.DataFrame({"num": [1.0, None], "cat": ["a", None]})
    test = pd.DataFrame({"num": [None], "cat": ["new"]})
    train_before = train.copy(deep=True)
    test_before = test.copy(deep=True)
    preprocessor = make_preprocessor()

    preprocessor.fit(train)
    preprocessor.transform(test)

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)


def test_metadata_serialization_has_no_raw_rows_or_objects() -> None:
    preprocessor = make_preprocessor().fit(pd.DataFrame({"num": [1.0, None], "cat": ["a", "b"]}))

    metadata = preprocessor.get_metadata()

    json.dumps(metadata, allow_nan=False)
    forbidden = {"raw_rows", "target", "predictions", "source_path"}
    assert not forbidden & set(metadata)


def test_metadata_does_not_change_after_transform() -> None:
    train = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["b", "a", "a"]})
    validation = pd.DataFrame({"num": [None, 9.0], "cat": ["new", "a"]})
    preprocessor = make_preprocessor().fit(train)
    before = preprocessor.get_metadata()

    preprocessor.transform(validation)
    after = preprocessor.get_metadata()

    assert before == after


def test_multiple_transforms_do_not_change_fitted_state() -> None:
    train = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["b", "a", "a"]})
    no_unseen = pd.DataFrame({"num": [None], "cat": ["a"]})
    one_unseen = pd.DataFrame({"num": [4.0], "cat": ["x"]})
    many_unseen = pd.DataFrame({"num": [5.0, None, 7.0], "cat": ["x", "y", "z"]})
    preprocessor = make_preprocessor().fit(train)
    expected_metadata = preprocessor.get_metadata()

    for frame in (no_unseen, one_unseen, many_unseen, no_unseen):
        preprocessor.transform(frame)
        assert preprocessor.get_metadata() == expected_metadata


def test_transform_order_does_not_affect_state_or_results() -> None:
    train = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["b", "a", "a"]})
    validation_1 = pd.DataFrame({"num": [None, 9.0], "cat": ["a", "new_1"]})
    validation_2 = pd.DataFrame({"num": [5.0, None], "cat": ["new_2", "b"]})
    first = make_preprocessor().fit(train)
    second = make_preprocessor().fit(train)

    first_v1 = first.transform(validation_1)
    first_v2 = first.transform(validation_2)
    second_v2 = second.transform(validation_2)
    second_v1 = second.transform(validation_1)

    assert first.get_metadata() == second.get_metadata()
    pd.testing.assert_frame_equal(first_v1, second_v1)
    pd.testing.assert_frame_equal(first_v2, second_v2)


def test_determinism_for_same_input_and_config() -> None:
    train = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["b", "a", "a"]})
    test = pd.DataFrame({"num": [None, 8.0], "cat": ["z", "b"]})
    first = make_preprocessor().fit(train)
    second = make_preprocessor().fit(train)

    pd.testing.assert_frame_equal(first.transform(test), second.transform(test))
    assert first.get_metadata() == second.get_metadata()


def test_reserved_token_collision_fails_fast() -> None:
    with pytest.raises(PreprocessingError, match="reserved unknown token"):
        make_preprocessor().fit(pd.DataFrame({"num": [1.0], "cat": ["__UNKNOWN__"]}))
    preprocessor = make_preprocessor().fit(pd.DataFrame({"num": [1.0], "cat": ["a"]}))
    with pytest.raises(PreprocessingError, match="reserved unknown token"):
        preprocessor.transform(pd.DataFrame({"num": [1.0], "cat": ["__UNKNOWN__"]}))


def test_protocol_config_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "protocols" / "protocol_a.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(protocol_payload(), sort_keys=False), encoding="utf-8")

    config = load_protocol_a_config(config_path, repo_root=tmp_path)

    assert config.numeric_imputation_strategy == "mean"
    payload = protocol_payload()
    payload["protocol"]["numeric_imputation"]["strategy"] = "median"
    with pytest.raises(PreprocessingError, match="Unsupported numeric"):
        parse_protocol_a_config(payload)


def test_feature_routing_from_loaded_dataset_metadata(tmp_path: Path) -> None:
    data_path = tmp_path / "data" / "raw" / "toy.csv"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ID": 1, "num": 1.0, "cat": "a", "BAD": 0},
            {"ID": 2, "num": 2.0, "cat": "b", "BAD": 1},
        ]
    ).to_csv(data_path, index=False)
    registry = {
        "datasets": {
            "toy": {
                "id": "toy",
                "active_file": "data/raw/toy.csv",
                "raw_file": "data/raw/toy.csv",
                "reader": {"type": "csv", "header": True},
                "target": {"column": "BAD", "mapping_to_binary": {0: 0, 1: 1}},
                "identifier_columns": ["ID"],
                "ignored_columns": ["ID"],
                "numeric_columns": ["num"],
                "categorical_columns": ["cat"],
                "missing_values": [],
            }
        }
    }
    registry_path = tmp_path / "data" / "datasets.yaml"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    loaded = load_dataset("toy", repo_root=tmp_path)

    preprocessor = LeakageSafePreprocessor(dataset_metadata=loaded.metadata).fit(loaded.features)

    assert list(loaded.features.columns) == ["num", "cat"]
    assert preprocessor.get_metadata()["numeric_features"] == ["num"]
    assert preprocessor.get_metadata()["categorical_features"] == ["cat"]


def test_gc_registry_metadata_routes_numeric_and_categorical_features() -> None:
    registry = load_registry()
    gc = registry["gc"]
    feature_columns = [*gc.numeric_columns, *gc.categorical_columns]
    metadata = {
        "target_column": gc.target_column,
        "removed_columns": [*gc.identifier_columns, *gc.ignored_columns],
        "numeric_columns": list(gc.numeric_columns),
        "categorical_columns": list(gc.categorical_columns),
    }

    numeric, categorical = features_from_dataset_metadata(feature_columns, metadata)

    assert numeric == list(gc.numeric_columns)
    assert categorical == list(gc.categorical_columns)
    assert gc.target_column not in set(numeric) | set(categorical)
