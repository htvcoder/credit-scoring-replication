from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.nested_cv import create_nested_cv_artifact, validate_nested_cv_artifact
from creditrep.checksums import get_dataset_checksum
from creditrep.config.exceptions import ConfigError
from creditrep.config.loader import sha256_canonical
from creditrep.config.nested import NestedCVConfig, parse_nested_cv_config
from creditrep.datasets.loader import load_dataset
from creditrep.datasets.models import LoadedDataset
from creditrep.experiments.nested_cv import FakeCandidateEstimator, run_nested_cv_validation
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.exceptions import SplitError
from creditrep.splitting.nested import create_nested_cv_definition, derive_seed, validate_nested_cv_definition

ROOT = Path(__file__).resolve().parents[1]


def toy_dataset(*, test_shift: float = 0.0, test_label_flip: bool = False) -> LoadedDataset:
    rows = []
    target = []
    for i in range(12):
        rows.append({"num": float(i), "cat": "a" if i % 2 == 0 else "b"})
        target.append(0)
    for i in range(12):
        rows.append({"num": float(100 + i), "cat": "c" if i % 2 == 0 else "d"})
        target.append(1)
    features = pd.DataFrame(rows)
    y = pd.Series(target, name="BAD")
    if test_shift:
        # This mutates rows that are outer test for at least one fold, but tests
        # compare inner state for a fixed fold where these rows are not used.
        features.loc[0:5, "num"] = features.loc[0:5, "num"] + test_shift
    if test_label_flip:
        y.loc[0:5] = 1 - y.loc[0:5]
    return LoadedDataset(
        dataset_id="toy",
        features=features,
        target=y,
        metadata={
            "dataset_id": "toy",
            "source_file": "data/raw/toy.csv",
            "target_column": "BAD",
            "removed_columns": [],
            "removed_identifier_columns": [],
            "removed_ignored_columns": [],
            "row_count": len(y),
            "feature_count": features.shape[1],
            "class_counts": {0: int((y == 0).sum()), 1: int((y == 1).sum())},
            "numeric_columns": ["num"],
            "categorical_columns": ["cat"],
        },
        source_path=Path("data/raw/toy.csv"),
    )


def nested_config(tmp_path: Path | None = None) -> NestedCVConfig:
    return NestedCVConfig(
        experiment_name="toy_nested_cv",
        dataset_id="TOY",
        output_root="artifacts/experiments" if tmp_path is None else "artifacts",
        protocol_config_path="configs/protocols/protocol_a.yaml",
        outer_strategy="repeated_stratified_2fold",
        outer_n_repeats=1,
        outer_n_splits=2,
        outer_shuffle=True,
        outer_random_seed=42,
        inner_strategy="stratified_kfold",
        inner_n_splits=2,
        inner_shuffle=True,
        candidates=({"name": "first", "bias": 0.0}, {"name": "second", "bias": 0.0}),
    )


def protocol_config() -> ProtocolAConfig:
    return ProtocolAConfig(woe_enabled=True, vif_enabled=False, scaling_enabled=True)


def make_definition(dataset: LoadedDataset | None = None):
    data = dataset or toy_dataset()
    return create_nested_cv_definition(
        data,
        dataset_checksum="ABC123",
        outer_n_repeats=1,
        inner_n_splits=2,
        random_seed=42,
    )


def test_outer_repeated_twofold_is_deterministic_and_covers_rows_once_per_repeat() -> None:
    dataset = toy_dataset()
    first = make_definition(dataset)
    second = make_definition(dataset)

    assert first.nested_cv_hash == second.nested_cv_hash
    assert [fold.outer_fold_id for fold in first.outer_folds] == ["repeat_00_fold_00", "repeat_00_fold_01"]
    for outer in first.outer_folds:
        assert not set(outer.train_indices) & set(outer.test_indices)
        assert set(outer.train_indices) | set(outer.test_indices) == set(range(24))
        assert outer.train_class_counts == {0: 6, 1: 6}
        assert outer.test_class_counts == {0: 6, 1: 6}
    test_rows = sorted(row for outer in first.outer_folds for row in outer.test_indices)
    assert test_rows == list(range(24))
    validate_nested_cv_definition(first, dataset.target)


def test_nested_cv_hash_changes_with_seed_and_seed_derivation_is_stable() -> None:
    dataset = toy_dataset()
    first = make_definition(dataset)
    changed = create_nested_cv_definition(dataset, dataset_checksum="ABC123", outer_n_repeats=1, inner_n_splits=2, random_seed=7)

    assert first.nested_cv_hash != changed.nested_cv_hash
    assert derive_seed(42, stage="inner", repeat_index=0, outer_fold_index=1) == derive_seed(
        42, stage="inner", repeat_index=0, outer_fold_index=1
    )
    assert derive_seed(42, stage="outer", repeat_index=0, outer_fold_index=0) != derive_seed(
        42, stage="inner", repeat_index=0, outer_fold_index=0
    )


def test_inner_folds_are_subset_of_outer_train_and_cover_validation_once() -> None:
    dataset = toy_dataset()
    definition = make_definition(dataset)
    for outer in definition.outer_folds:
        outer_train = set(outer.train_indices)
        outer_test = set(outer.test_indices)
        validation_rows = []
        for inner in outer.inner_folds:
            assert set(inner.train_indices) <= outer_train
            assert set(inner.validation_indices) <= outer_train
            assert not set(inner.train_indices) & set(inner.validation_indices)
            assert not outer_test & set(inner.train_indices)
            assert not outer_test & set(inner.validation_indices)
            validation_rows.extend(inner.validation_indices)
        assert sorted(validation_rows) == sorted(outer.train_indices)


def test_insufficient_class_count_fails_fast() -> None:
    dataset = LoadedDataset(
        dataset_id="tiny",
        features=pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]}),
        target=pd.Series([0, 0, 1], name="BAD"),
        metadata={"source_file": "data/raw/tiny.csv"},
        source_path=Path("data/raw/tiny.csv"),
    )
    with pytest.raises(SplitError, match="minority class"):
        create_nested_cv_definition(dataset, dataset_checksum="ABC", inner_n_splits=2)


def test_config_validation_rejects_bad_nested_cv_payload() -> None:
    payload = nested_config().canonical_payload()
    parsed = parse_nested_cv_config(payload)
    assert parsed.publishable is False
    payload["experiment"]["publishable"] = True
    with pytest.raises(ConfigError, match="publishable"):
        parse_nested_cv_config(payload)


def test_per_fold_preprocessing_is_fresh_numeric_finite_and_metadata_stable() -> None:
    dataset = toy_dataset()
    definition = make_definition(dataset)
    result = run_nested_cv_validation(
        config=nested_config(),
        dataset=dataset,
        nested_cv=definition,
        protocol_config=protocol_config(),
    )

    ids = list(result.pipeline_instance_ids.values())
    assert len(ids) == len(set(ids))
    for metadata in result.inner_preprocessing.values():
        json.dumps(metadata, allow_nan=False)
        output_order = metadata["preprocessing"]["final_output_feature_order"]
        assert output_order
    for metadata in result.outer_preprocessing.values():
        assert metadata["preprocessing"]["fitted_row_count"] == 12


def test_outer_test_labels_and_features_do_not_change_first_outer_inner_state_or_selected_candidate() -> None:
    base = toy_dataset()
    definition = make_definition(base)
    outer = definition.outer_folds[0]
    mutated_features = base.features.copy(deep=True)
    mutated_target = base.target.copy(deep=True)
    mutated_features.iloc[list(outer.test_indices), mutated_features.columns.get_loc("num")] = 999999.0
    mutated_target.iloc[list(outer.test_indices)] = 1 - mutated_target.iloc[list(outer.test_indices)]
    mutated = replace(base, features=mutated_features, target=mutated_target)

    base_result = run_nested_cv_validation(
        config=nested_config(),
        dataset=base,
        nested_cv=definition,
        protocol_config=protocol_config(),
    )
    mutated_result = run_nested_cv_validation(
        config=nested_config(),
        dataset=mutated,
        nested_cv=definition,
        protocol_config=protocol_config(),
    )

    for inner in outer.inner_folds:
        assert base_result.inner_preprocessing[inner.inner_fold_id] == mutated_result.inner_preprocessing[inner.inner_fold_id]
    assert (
        base_result.tuning_summaries[outer.outer_fold_id]["selected_candidate_index"]
        == mutated_result.tuning_summaries[outer.outer_fold_id]["selected_candidate_index"]
    )
    assert base_result.tuning_summaries[outer.outer_fold_id]["outer_test_metric"] is None


def test_tuning_uses_fresh_estimators_and_tie_breaks_by_candidate_order() -> None:
    FakeCandidateEstimator.fit_counter = 0
    dataset = toy_dataset()
    definition = make_definition(dataset)
    result = run_nested_cv_validation(
        config=nested_config(),
        dataset=dataset,
        nested_cv=definition,
        protocol_config=protocol_config(),
    )
    summary = result.tuning_summaries["repeat_00_fold_00"]
    fit_ids = [fit_id for candidate in summary["candidate_results"] for fit_id in candidate["estimator_fit_ids"]]

    assert len(fit_ids) == len(set(fit_ids))
    assert summary["selected_candidate_index"] == 0


def test_artifact_round_trip_no_overwrite_and_corruption_detection(tmp_path: Path) -> None:
    dataset = toy_dataset()
    definition = make_definition(dataset)
    result = run_nested_cv_validation(
        config=nested_config(tmp_path),
        dataset=dataset,
        nested_cv=definition,
        protocol_config=protocol_config(),
    )
    checksum = type("Checksum", (), {"actual_sha256": "ABC123", "declared_sha256": "ABC123", "matches": True})()
    artifact_dir, manifest = create_nested_cv_artifact(
        config=nested_config(tmp_path),
        protocol_config=protocol_config(),
        dataset=dataset,
        checksum=checksum,
        result=result,
        repo_root=tmp_path,
        created_at_utc=datetime(2026, 7, 22, 1, 2, 3, tzinfo=timezone.utc),
    )

    assert manifest["publishable"] is False
    assert manifest["result_scope"] == "preprocessing_validation"
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in artifact_dir.rglob("*.json"))
    assert "predictions" in artifact_text
    assert "999999" not in artifact_text
    validate_nested_cv_artifact(artifact_dir, target=dataset.target)
    with pytest.raises(ArtifactError, match="overwritten"):
        create_nested_cv_artifact(
            config=nested_config(tmp_path),
            protocol_config=protocol_config(),
            dataset=dataset,
            checksum=checksum,
            result=result,
            repo_root=tmp_path,
            created_at_utc=datetime(2026, 7, 22, 1, 2, 3, tzinfo=timezone.utc),
        )
    folds_path = artifact_dir / "nested_cv" / "outer_folds.json"
    payload = json.loads(folds_path.read_text(encoding="utf-8"))
    payload["outer_folds"][0]["test_indices"] = payload["outer_folds"][0]["test_indices"][:-1]
    folds_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="hash mismatch|does not cover"):
        validate_nested_cv_artifact(artifact_dir, target=dataset.target)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def prepare_cli_repo(tmp_path: Path) -> Path:
    rows = []
    for i in range(8):
        rows.append({"num": float(i), "cat": "a" if i % 2 == 0 else "b", "BAD": 0})
    for i in range(8):
        rows.append({"num": float(100 + i), "cat": "c" if i % 2 == 0 else "d", "BAD": 1})
    data_path = tmp_path / "data" / "raw" / "toy.csv"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(data_path, index=False)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest().upper()
    write_yaml(
        tmp_path / "data" / "datasets.yaml",
        {
            "datasets": {
                "toy": {
                    "id": "toy",
                    "active_file": "data/raw/toy.csv",
                    "raw_file": "data/raw/toy.csv",
                    "reader": {"type": "csv", "header": True},
                    "target": {"column": "BAD", "mapping_to_binary": {0: 0, 1: 1}},
                    "identifier_columns": [],
                    "ignored_columns": [],
                    "numeric_columns": ["num"],
                    "categorical_columns": ["cat"],
                    "missing_values": [],
                }
            }
        },
    )
    (tmp_path / "data" / "checksums-sha256.csv").write_text(
        f'"Path","Algorithm","Hash"\n"data/raw/toy.csv","SHA256","{digest}"\n',
        encoding="utf-8",
    )
    write_yaml(
        tmp_path / "configs" / "protocols" / "protocol_a.yaml",
        {
            "protocol": {
                "name": "protocol_a",
                "version": "p3b-v1",
                "numeric_imputation": {"strategy": "mean"},
                "categorical_imputation": {"strategy": "most_frequent"},
                "unseen_category": {"strategy": "reserved_token", "token": "__UNKNOWN__"},
                "woe": {"enabled": True, "scope": "categorical", "smoothing": 0.5, "unknown_value": 0.0},
                "vif": {"enabled": False, "threshold": 10.0, "minimum_features_to_keep": 1},
                "scaling": {"enabled": False, "strategy": "standard"},
            }
        },
    )
    config_path = tmp_path / "configs" / "experiments" / "nested_cv_toy.yaml"
    write_yaml(
        config_path,
        {
            "experiment": {"name": "nested_cv_toy", "result_scope": "preprocessing_validation", "publishable": False},
            "dataset": {"id": "TOY"},
            "cross_validation": {
                "outer": {
                    "strategy": "repeated_stratified_2fold",
                    "n_repeats": 1,
                    "n_splits": 2,
                    "shuffle": True,
                    "random_seed": 42,
                },
                "inner": {
                    "strategy": "stratified_kfold",
                    "n_splits": 2,
                    "shuffle": True,
                    "random_seed_policy": "derived_from_outer",
                },
            },
            "preprocessing": {"protocol_config": "configs/protocols/protocol_a.yaml"},
            "tuning": {"candidates": [{"name": "a", "bias": 0.0}, {"name": "b", "bias": 0.1}]},
            "output": {"root_dir": "artifacts/experiments"},
        },
    )
    return config_path


def test_cli_creates_reduced_nested_cv_artifact_and_rejects_overwrite(tmp_path: Path) -> None:
    config_path = prepare_cli_repo(tmp_path)
    script = ROOT / "scripts" / "create_nested_cv_artifact.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["publishable"] is False
    assert summary["result_scope"] == "preprocessing_validation"
    artifact_dir = tmp_path / summary["artifact_directory"]
    assert (artifact_dir / "nested_cv" / "outer_folds.json").exists()
    assert "num" not in (artifact_dir / "manifest.json").read_text(encoding="utf-8")
