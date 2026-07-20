from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditrep.artifacts.exceptions import ArtifactError
from creditrep.artifacts.split_definition import load_split_csv, validate_split_definition
from creditrep.artifacts.writer import create_split_artifact
from creditrep.checksums import get_dataset_checksum
from creditrep.config.exceptions import ConfigError
from creditrep.config.loader import config_hash, load_experiment_config, parse_experiment_config
from creditrep.datasets import load_dataset
from creditrep.datasets.models import LoadedDataset
from creditrep.splitting import create_split
from creditrep.splitting.exceptions import SplitError

ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def valid_config(output_root: str = "artifacts/experiments") -> dict:
    return {
        "experiment": {"name": "toy_split_validation"},
        "dataset": {"id": "TOY"},
        "split": {
            "strategy": "stratified_holdout",
            "test_size": 0.25,
            "random_seed": 42,
            "shuffle": True,
        },
        "output": {"root_dir": output_root},
    }


def toy_loaded_dataset(n_per_class: int = 20, *, checksum: str = "ABC123") -> LoadedDataset:
    rows = []
    target = []
    for i in range(n_per_class):
        rows.append({"score": i, "bucket": "good"})
        target.append(0)
    for i in range(n_per_class):
        rows.append({"score": 100 + i, "bucket": "bad"})
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
            "row_count": len(y),
            "feature_count": features.shape[1],
            "class_counts": {0: n_per_class, 1: n_per_class},
            "default_rate": 0.5,
            "checksum_sha256": checksum,
        },
        source_path=Path("data/raw/toy.csv"),
    )


def prepare_toy_repo(tmp_path: Path) -> tuple[Path, str]:
    rows = [{"score": i, "BAD": 0} for i in range(12)] + [{"score": 100 + i, "BAD": 1} for i in range(12)]
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
                    "full_name": "Toy",
                    "active_file": "data/raw/toy.csv",
                    "raw_file": "data/raw/toy.csv",
                    "reader": {"type": "csv", "header": True},
                    "target": {"column": "BAD", "mapping_to_binary": {0: 0, 1: 1}},
                    "identifier_columns": [],
                    "ignored_columns": [],
                    "categorical_columns": [],
                    "numeric_columns": ["score"],
                    "missing_values": [],
                }
            }
        },
    )
    checksum_path = tmp_path / "data" / "checksums-sha256.csv"
    checksum_path.write_text(f'"Path","Algorithm","Hash"\n"data/raw/toy.csv","SHA256","{digest}"\n', encoding="utf-8")
    return data_path, digest


def split_hash_payload(dataset: LoadedDataset, split, checksum: str, config) -> dict:
    return {
        "dataset": {
            "checksum_sha256": checksum,
            "id": dataset.dataset_id,
            "source_file": dataset.metadata["source_file"],
        },
        "split": {
            "random_seed": config.random_seed,
            "strategy": config.split_strategy,
            "test_indices": list(split.test_indices),
            "test_size": config.test_size,
            "train_indices": list(split.train_indices),
        },
    }


def test_config_loads_successfully(tmp_path):
    config_path = tmp_path / "configs" / "split.yaml"
    write_yaml(config_path, valid_config())

    config = load_experiment_config(config_path, repo_root=tmp_path)

    assert config.dataset_id == "TOY"
    assert config.split_strategy == "stratified_holdout"
    assert config.test_size == 0.25


def test_config_file_missing_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_experiment_config(tmp_path / "missing.yaml", repo_root=tmp_path)


def test_invalid_yaml_is_rejected(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("experiment: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML is invalid"):
        load_experiment_config(config_path, repo_root=tmp_path)


def test_unsupported_strategy_is_rejected():
    payload = valid_config()
    payload["split"]["strategy"] = "random_holdout"

    with pytest.raises(ConfigError, match="Unsupported split strategy"):
        parse_experiment_config(payload)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.2])
def test_invalid_test_size_is_rejected(test_size):
    payload = valid_config()
    payload["split"]["test_size"] = test_size

    with pytest.raises(ConfigError, match="test_size"):
        parse_experiment_config(payload)


def test_invalid_seed_is_rejected():
    payload = valid_config()
    payload["split"]["random_seed"] = "42"

    with pytest.raises(ConfigError, match="random_seed"):
        parse_experiment_config(payload)


def test_config_output_path_must_be_portable():
    payload = valid_config("D:/tmp/artifacts")

    with pytest.raises(ConfigError, match="absolute path"):
        parse_experiment_config(payload)


def test_config_hash_is_stable_across_key_order():
    left = parse_experiment_config(valid_config())
    right_payload = {
        "output": {"root_dir": "artifacts/experiments"},
        "split": {"shuffle": True, "random_seed": 42, "test_size": 0.25, "strategy": "stratified_holdout"},
        "dataset": {"id": "TOY"},
        "experiment": {"name": "toy_split_validation"},
    }
    right = parse_experiment_config(right_payload)

    assert config_hash(left) == config_hash(right)


def test_stratified_split_counts_no_overlap_and_no_lost_rows():
    dataset = toy_loaded_dataset()
    split = create_split(dataset, test_size=0.25, random_seed=42)

    assert len(split.train_indices) == 30
    assert len(split.test_indices) == 10
    assert not set(split.train_indices) & set(split.test_indices)
    assert set(split.train_indices) | set(split.test_indices) == set(range(40))
    assert split.metadata["train_class_counts"] == {0: 15, 1: 15}
    assert split.metadata["test_class_counts"] == {0: 5, 1: 5}


def test_same_seed_creates_same_indices_and_hash():
    dataset = toy_loaded_dataset()

    first = create_split(dataset, test_size=0.25, random_seed=42)
    second = create_split(dataset, test_size=0.25, random_seed=42)

    assert first.train_indices == second.train_indices
    assert first.test_indices == second.test_indices
    assert first.split_hash == second.split_hash


def test_different_seed_changes_split_and_hash():
    dataset = toy_loaded_dataset()

    first = create_split(dataset, test_size=0.25, random_seed=42)
    second = create_split(dataset, test_size=0.25, random_seed=43)

    assert first.test_indices != second.test_indices
    assert first.split_hash != second.split_hash


def test_split_hash_changes_when_assignment_changes():
    dataset = toy_loaded_dataset()
    split = create_split(dataset, test_size=0.25, random_seed=42)
    mutated = {
        "dataset": {"id": "toy", "checksum_sha256": "ABC123", "source_file": "data/raw/toy.csv"},
        "split": {
            "strategy": "stratified_holdout",
            "test_size": 0.25,
            "random_seed": 42,
            "train_indices": list(split.test_indices),
            "test_indices": list(split.train_indices),
        },
    }

    from creditrep.splitting.hashing import split_hash

    assert split_hash(mutated) != split.split_hash


def test_single_class_target_is_rejected():
    dataset = toy_loaded_dataset()
    bad = LoadedDataset(
        dataset_id="toy",
        features=dataset.features,
        target=pd.Series([0] * len(dataset.target), name="BAD"),
        metadata=dataset.metadata,
        source_path=dataset.source_path,
    )

    with pytest.raises(SplitError, match="classes"):
        create_split(bad)


def test_too_small_class_is_rejected():
    dataset = LoadedDataset(
        dataset_id="toy",
        features=pd.DataFrame({"score": list(range(11))}),
        target=pd.Series([0] * 10 + [1], name="BAD"),
        metadata={"source_file": "data/raw/toy.csv", "checksum_sha256": "ABC123"},
        source_path=Path("data/raw/toy.csv"),
    )

    with pytest.raises(SplitError, match="too few"):
        create_split(dataset)


def test_duplicate_index_is_rejected():
    dataset = toy_loaded_dataset()
    features = dataset.features.copy()
    target = dataset.target.copy()
    features.index = [0] * len(features)
    target.index = [0] * len(target)
    duplicate = LoadedDataset("toy", features, target, dataset.metadata, dataset.source_path)

    with pytest.raises(SplitError, match="duplicate indices"):
        create_split(duplicate)


def test_features_target_length_mismatch_is_rejected():
    dataset = toy_loaded_dataset()
    mismatch = LoadedDataset(
        "toy",
        dataset.features.iloc[:-1],
        dataset.target,
        dataset.metadata,
        dataset.source_path,
    )

    with pytest.raises(SplitError, match="length mismatch"):
        create_split(mismatch)


def test_checksum_match_and_mismatch(tmp_path):
    _, digest = prepare_toy_repo(tmp_path)

    status = get_dataset_checksum("toy", "data/raw/toy.csv", repo_root=tmp_path)
    assert status.actual_sha256 == digest
    assert status.matches

    checksum_path = tmp_path / "data" / "checksums-sha256.csv"
    checksum_path.write_text('"Path","Algorithm","Hash"\n"data/raw/toy.csv","SHA256","BAD"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        get_dataset_checksum("toy", "data/raw/toy.csv", repo_root=tmp_path)


def test_artifact_manifest_and_split_definition_round_trip(tmp_path):
    prepare_toy_repo(tmp_path)
    config = parse_experiment_config(valid_config())
    dataset = load_dataset("TOY", repo_root=tmp_path)
    checksum = get_dataset_checksum("toy", "data/raw/toy.csv", repo_root=tmp_path)
    dataset.metadata["checksum_sha256"] = checksum.actual_sha256
    split = create_split(dataset, test_size=config.test_size, random_seed=config.random_seed)
    artifact_dir = create_split_artifact(
        config=config,
        dataset=dataset,
        split=split,
        checksum=checksum,
        repo_root=tmp_path,
        created_at_utc=datetime(2026, 7, 20, 1, 2, 3, tzinfo=timezone.utc),
    )

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    split_json = json.loads((artifact_dir / "split.json").read_text(encoding="utf-8"))
    assignments = load_split_csv(artifact_dir / "split.csv")

    assert manifest["dataset"]["checksum_sha256"] == checksum.actual_sha256
    assert manifest["provenance"]["config_hash"] == config_hash(config)
    assert manifest["split"]["split_hash"] == split.split_hash
    assert manifest["provenance"]["git_available"] in {True, False}
    assert set(assignments.values()) == {"train", "test"}
    validate_split_definition(
        artifact_dir / "split.csv",
        row_count=split.metadata["row_count"],
        expected_split_hash=split.split_hash,
        split_hash_payload=split_json["split_hash_payload"],
    )
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in artifact_dir.glob("*.json"))
    assert "D:\\" not in artifact_text
    assert "score" not in manifest
    assert "BAD" not in manifest


def test_artifact_does_not_overwrite_existing_directory(tmp_path):
    prepare_toy_repo(tmp_path)
    config = parse_experiment_config(valid_config())
    dataset = load_dataset("TOY", repo_root=tmp_path)
    checksum = get_dataset_checksum("toy", "data/raw/toy.csv", repo_root=tmp_path)
    dataset.metadata["checksum_sha256"] = checksum.actual_sha256
    split = create_split(dataset, test_size=config.test_size, random_seed=config.random_seed)
    created = datetime(2026, 7, 20, 1, 2, 3, tzinfo=timezone.utc)
    create_split_artifact(
        config=config,
        dataset=dataset,
        split=split,
        checksum=checksum,
        repo_root=tmp_path,
        created_at_utc=created,
    )

    with pytest.raises(ArtifactError, match="will not be overwritten"):
        create_split_artifact(
            config=config,
            dataset=dataset,
            split=split,
            checksum=checksum,
            repo_root=tmp_path,
            created_at_utc=created,
        )


def test_tampered_split_definition_is_detected(tmp_path):
    prepare_toy_repo(tmp_path)
    config = parse_experiment_config(valid_config())
    dataset = load_dataset("TOY", repo_root=tmp_path)
    checksum = get_dataset_checksum("toy", "data/raw/toy.csv", repo_root=tmp_path)
    dataset.metadata["checksum_sha256"] = checksum.actual_sha256
    split = create_split(dataset, test_size=config.test_size, random_seed=config.random_seed)
    artifact_dir = create_split_artifact(
        config=config,
        dataset=dataset,
        split=split,
        checksum=checksum,
        repo_root=tmp_path,
        created_at_utc=datetime(2026, 7, 20, 1, 2, 3, tzinfo=timezone.utc),
    )
    split_json = json.loads((artifact_dir / "split.json").read_text(encoding="utf-8"))
    split_csv = artifact_dir / "split.csv"
    lines = split_csv.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("train", "test") if "train" in lines[1] else lines[1].replace("test", "train")
    split_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="Split hash mismatch"):
        validate_split_definition(
            split_csv,
            row_count=split.metadata["row_count"],
            expected_split_hash=split.split_hash,
            split_hash_payload=split_json["split_hash_payload"],
        )


def test_cli_creates_artifact_and_invalid_config_exits_nonzero(tmp_path):
    prepare_toy_repo(tmp_path)
    config_path = tmp_path / "configs" / "experiments" / "split_toy.yaml"
    write_yaml(config_path, valid_config())
    script = ROOT / "scripts" / "create_split_artifact.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["dataset"] == "TOY"
    assert summary["train_rows"] == 18
    assert summary["test_rows"] == 6
    assert (tmp_path / summary["experiment_directory"] / "manifest.json").exists()

    bad_config = tmp_path / "bad.yaml"
    write_yaml(bad_config, {**valid_config(), "split": {"strategy": "bad", "test_size": 0.2, "random_seed": 1}})
    failed = subprocess.run(
        [sys.executable, str(script), "--config", str(bad_config)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
