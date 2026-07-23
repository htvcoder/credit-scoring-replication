from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditrep.artifacts.metric_validation import (
    create_metric_validation_artifact,
    load_metric_validation_artifact,
    validate_metric_validation_artifact,
)
from creditrep.artifacts.exceptions import ArtifactError
from creditrep.config.exceptions import ConfigError
from creditrep.config.metric_validation import MetricValidationConfig, parse_metric_validation_config
from creditrep.datasets.models import LoadedDataset
from creditrep.experiments.metric_validation import run_metric_validation
from creditrep.metrics.registry import metric_config_hash
from creditrep.preprocessing import ProtocolAConfig
from creditrep.splitting.nested import create_nested_cv_definition

ROOT = Path(__file__).resolve().parents[1]


def toy_dataset() -> LoadedDataset:
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


def metric_validation_config(tmp_output: str = "artifacts") -> MetricValidationConfig:
    return MetricValidationConfig(
        experiment_name="toy_metric_validation",
        dataset_id="TOY",
        output_root=tmp_output,
        protocol_config_path="configs/protocols/protocol_a.yaml",
        outer_strategy="repeated_stratified_2fold",
        outer_n_repeats=1,
        outer_n_splits=2,
        outer_shuffle=True,
        outer_random_seed=42,
        inner_strategy="stratified_kfold",
        inner_n_splits=2,
        inner_shuffle=True,
        candidates=(
            {"name": "low_bias", "bias": -0.1},
            {"name": "neutral_bias", "bias": 0.0},
            {"name": "high_bias", "bias": 0.1},
        ),
        metrics=parse_metric_validation_config(
            {
                "experiment": {"name": "toy_metric_validation", "result_scope": "metric_validation", "publishable": False},
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
                "evaluation": {
                    "validation_model": "deterministic_probability_estimator",
                    "metrics": [
                        {"id": "roc_auc"},
                        {"id": "brier_score"},
                        {"id": "partial_gini"},
                        {"id": "emp"},
                    ],
                },
                "tuning": {"candidates": [{"name": "seed", "bias": 0.0}]},
                "output": {"root_dir": "artifacts"},
            }
        ).metrics,
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


def test_metric_config_validation_defaults_hash_and_rejections() -> None:
    payload = {
        "experiment": {"name": "metric_validation", "result_scope": "metric_validation", "publishable": False},
        "dataset": {"id": "GC"},
        "cross_validation": {
            "outer": {
                "strategy": "repeated_stratified_2fold",
                "n_repeats": 1,
                "n_splits": 2,
                "shuffle": True,
                "random_seed": 42,
            },
            "inner": {"strategy": "stratified_kfold", "n_splits": 2, "shuffle": True, "random_seed_policy": "derived_from_outer"},
        },
        "preprocessing": {"protocol_config": "configs/protocols/protocol_a.yaml"},
        "evaluation": {
            "validation_model": "deterministic_probability_estimator",
            "metrics": [{"id": "roc_auc"}, {"id": "partial_gini"}, {"id": "emp"}],
        },
        "tuning": {"candidates": [{"name": "a", "bias": 0.0}]},
        "output": {"root_dir": "artifacts/experiments"},
    }

    parsed = parse_metric_validation_config(payload)

    assert [metric.metric_id for metric in parsed.metrics] == ["roc_auc", "partial_gini", "emp"]
    assert parsed.metrics[1].parameters == {"b": 0.4}
    assert metric_config_hash(parsed.metrics) == metric_config_hash(parsed.metrics)
    assert parsed.canonical_payload() == parsed.canonical_payload()

    duplicate = replace(parsed, metrics=parsed.metrics + (parsed.metrics[0],))
    with pytest.raises(TypeError):
        json.dumps(duplicate)

    bad_unknown_metric = payload | {"evaluation": {"validation_model": "deterministic_probability_estimator", "metrics": [{"id": "unknown"}]}}
    with pytest.raises(ConfigError, match="Unsupported metric id"):
        parse_metric_validation_config(bad_unknown_metric)

    bad_duplicate = payload | {
        "evaluation": {"validation_model": "deterministic_probability_estimator", "metrics": [{"id": "roc_auc"}, {"id": "roc_auc"}]}
    }
    with pytest.raises(ConfigError, match="Duplicate metric id"):
        parse_metric_validation_config(bad_duplicate)

    bad_partial = payload | {
        "evaluation": {
            "validation_model": "deterministic_probability_estimator",
            "metrics": [{"id": "partial_gini", "parameters": {"b": 1.0}}],
        }
    }
    with pytest.raises(ConfigError, match="must be > 0 and < 1"):
        parse_metric_validation_config(bad_partial)

    bad_emp = payload | {
        "evaluation": {
            "validation_model": "deterministic_probability_estimator",
            "metrics": [{"id": "emp", "parameters": {"b1": 1.0}}],
        }
    }
    with pytest.raises(ConfigError, match="unsupported parameters"):
        parse_metric_validation_config(bad_emp)


def test_metric_validation_uses_outer_test_only_for_final_metric_evaluation() -> None:
    base = toy_dataset()
    definition = make_definition(base)
    outer = definition.outer_folds[0]
    mutated_features = base.features.copy(deep=True)
    mutated_target = base.target.copy(deep=True)
    mutated_features.iloc[list(outer.test_indices), mutated_features.columns.get_loc("num")] = 999999.0
    mutated_target.iloc[list(outer.test_indices)] = 1 - mutated_target.iloc[list(outer.test_indices)]
    mutated = replace(base, features=mutated_features, target=mutated_target)

    base_result = run_metric_validation(
        config=metric_validation_config(),
        dataset=base,
        nested_cv=definition,
        protocol_config=protocol_config(),
    )
    mutated_result = run_metric_validation(
        config=metric_validation_config(),
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
    base_fold = {metric.metric_id: metric for metric in base_result.fold_metrics[outer.outer_fold_id]}
    mutated_fold = {metric.metric_id: metric for metric in mutated_result.fold_metrics[outer.outer_fold_id]}
    assert base_fold["roc_auc"].value != mutated_fold["roc_auc"].value
    assert base_fold["brier_score"].value != mutated_fold["brier_score"].value
    assert base_fold["emp"].status == "unsupported"
    assert mutated_fold["emp"].status == "unsupported"


def test_metric_validation_artifact_round_trip_and_cli_dry_run(tmp_path: Path) -> None:
    dataset = toy_dataset()
    definition = make_definition(dataset)
    config = metric_validation_config("artifacts")
    result = run_metric_validation(
        config=config,
        dataset=dataset,
        nested_cv=definition,
        protocol_config=protocol_config(),
    )
    checksum = type("Checksum", (), {"actual_sha256": "ABC123", "declared_sha256": "ABC123", "matches": True})()
    artifact_dir, manifest = create_metric_validation_artifact(
        config=config,
        protocol_config=protocol_config(),
        dataset=dataset,
        checksum=checksum,
        result=result,
        repo_root=tmp_path,
        created_at_utc=datetime(2026, 7, 23, 2, 3, 4, tzinfo=timezone.utc),
    )

    loaded = load_metric_validation_artifact(artifact_dir)
    assert manifest["publishable"] is False
    assert manifest["result_scope"] == "metric_validation"
    assert loaded["metrics_summary"]["metrics"]["emp"]["mean"] is None
    assert loaded["manifest"]["reserved_artifacts"]["predictions"] is None
    validate_metric_validation_artifact(artifact_dir, target=dataset.target)
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in artifact_dir.rglob("*.json"))
    assert "row_position" not in artifact_text
    assert "999999" not in artifact_text
    with pytest.raises(ArtifactError, match="overwritten"):
        create_metric_validation_artifact(
            config=config,
            protocol_config=protocol_config(),
            dataset=dataset,
            checksum=checksum,
            result=result,
            repo_root=tmp_path,
            created_at_utc=datetime(2026, 7, 23, 2, 3, 4, tzinfo=timezone.utc),
        )

    config_path = tmp_path / "configs" / "experiments" / "metric_validation_toy.yaml"
    data_path = tmp_path / "data" / "raw" / "toy.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame = dataset.features.copy(deep=True)
    frame["BAD"] = dataset.target
    frame.to_csv(data_path, index=False)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest().upper()
    (tmp_path / "data" / "checksums-sha256.csv").write_text(
        f'"Path","Algorithm","Hash"\n"data/raw/toy.csv","SHA256","{digest}"\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "configs" / "protocols").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "datasets.yaml").write_text(
        yaml.safe_dump(
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
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "protocols" / "protocol_a.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol": {
                    "name": "protocol_a",
                    "version": "p3b-v1",
                    "numeric_imputation": {"strategy": "mean"},
                    "categorical_imputation": {"strategy": "most_frequent"},
                    "unseen_category": {"strategy": "reserved_token", "token": "__UNKNOWN__"},
                    "woe": {"enabled": True, "scope": "categorical", "smoothing": 0.5, "unknown_value": 0.0},
                    "vif": {"enabled": False, "threshold": 10.0, "minimum_features_to_keep": 1},
                    "scaling": {"enabled": True, "strategy": "standard"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "metric_validation_toy", "result_scope": "metric_validation", "publishable": False},
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
                "evaluation": {
                    "validation_model": "deterministic_probability_estimator",
                    "metrics": [{"id": "roc_auc"}, {"id": "brier_score"}, {"id": "partial_gini"}, {"id": "emp"}],
                },
                "tuning": {"candidates": [{"name": "a", "bias": 0.0}, {"name": "b", "bias": 0.1}]},
                "output": {"root_dir": "artifacts/experiments"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_metric_validation.py"), "--config", str(config_path), "--dry-run"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["dry_run"] is True
    assert summary["metrics"] == ["roc_auc", "brier_score", "partial_gini", "emp"]
