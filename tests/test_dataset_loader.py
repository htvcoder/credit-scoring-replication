from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from creditrep.datasets.exceptions import (
    DatasetFileError,
    DatasetNotFoundError,
    DatasetSchemaError,
    RegistryError,
)
from creditrep.datasets.loader import load_dataset
from creditrep.datasets.registry import load_registry


def write_registry(root: Path, datasets: dict) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    registry_path = data_dir / "datasets.yaml"
    registry_path.write_text(yaml.safe_dump({"datasets": datasets}, sort_keys=False), encoding="utf-8")
    return registry_path


def base_spec(
    dataset_id: str,
    file_path: str,
    *,
    target_column: str = "BAD",
    mapping: dict[int, int] | None = None,
    identifiers: list[str] | None = None,
    ignored: list[str] | None = None,
) -> dict:
    return {
        "id": dataset_id,
        "active_file": file_path,
        "raw_file": file_path,
        "reader": {"type": "csv", "header": True},
        "target": {
            "column": target_column,
            "mapping_to_binary": mapping or {0: 0, 1: 1},
        },
        "identifier_columns": identifiers or [],
        "ignored_columns": ignored or [],
        "categorical_columns": [],
        "numeric_columns": ["score"],
        "missing_values": [],
    }


def write_csv(root: Path, relative: str, rows: list[dict]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_registry_loads_successfully(tmp_path):
    write_csv(tmp_path, "data/raw/toy.csv", [{"score": 10, "BAD": 0}, {"score": 20, "BAD": 1}])
    registry_path = write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})

    registry = load_registry(registry_path, repo_root=tmp_path)

    assert set(registry) == {"toy"}
    assert registry["toy"].active_file == "data/raw/toy.csv"


def test_unknown_dataset_id_is_rejected(tmp_path):
    write_csv(tmp_path, "data/raw/toy.csv", [{"score": 10, "BAD": 0}, {"score": 20, "BAD": 1}])
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})

    with pytest.raises(DatasetNotFoundError, match="missing"):
        load_dataset("missing", repo_root=tmp_path)


def test_missing_active_file_is_rejected(tmp_path):
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/missing.csv")})

    with pytest.raises(DatasetFileError, match="active file"):
        load_dataset("toy", repo_root=tmp_path)


def test_binary_target_mapping_is_preserved(tmp_path):
    write_csv(tmp_path, "data/raw/toy.csv", [{"score": 10, "BAD": 0}, {"score": 20, "BAD": 1}])
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})

    loaded = load_dataset("TOY", repo_root=tmp_path)

    assert loaded.target.tolist() == [0, 1]
    assert set(loaded.target.unique()) == {0, 1}


def test_gc_mapping_1_to_0_and_2_to_1(tmp_path):
    write_csv(tmp_path, "data/raw/gc.csv", [{"score": 10, "target": 1}, {"score": 20, "target": 2}])
    write_registry(
        tmp_path,
        {
            "gc": base_spec(
                "gc",
                "data/raw/gc.csv",
                target_column="target",
                mapping={1: 0, 2: 1},
            )
        },
    )

    loaded = load_dataset("GC", repo_root=tmp_path)

    assert loaded.target.tolist() == [0, 1]
    assert loaded.metadata["target_mapping"] == {"1": 0, "2": 1}


def test_identifier_and_target_columns_are_removed_from_features(tmp_path):
    write_csv(
        tmp_path,
        "data/raw/tc.csv",
        [{"ID": 1, "score": 10, "BAD": 0}, {"ID": 2, "score": 20, "BAD": 1}],
    )
    write_registry(tmp_path, {"tc": base_spec("tc", "data/raw/tc.csv", identifiers=["ID"], ignored=["ID"])})

    loaded = load_dataset("tc", repo_root=tmp_path)

    assert list(loaded.features.columns) == ["score"]
    assert "BAD" not in loaded.features.columns
    assert loaded.metadata["removed_identifier_columns"] == ["ID"]


def test_unknown_target_value_is_rejected(tmp_path):
    write_csv(tmp_path, "data/raw/toy.csv", [{"score": 10, "BAD": 0}, {"score": 20, "BAD": 2}])
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})

    with pytest.raises(DatasetSchemaError, match="outside mapping"):
        load_dataset("toy", repo_root=tmp_path)


def test_missing_target_column_is_rejected(tmp_path):
    write_csv(tmp_path, "data/raw/toy.csv", [{"score": 10}, {"score": 20}])
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})

    with pytest.raises(DatasetSchemaError, match="target column"):
        load_dataset("toy", repo_root=tmp_path)


def test_target_null_after_normalization_is_rejected(tmp_path):
    write_csv(tmp_path, "data/raw/toy.csv", [{"score": 10, "BAD": 0}, {"score": 20, "BAD": None}])
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})

    with pytest.raises(DatasetSchemaError, match="null values"):
        load_dataset("toy", repo_root=tmp_path)


def test_metadata_counts_are_correct(tmp_path):
    write_csv(
        tmp_path,
        "data/raw/toy.csv",
        [{"score": 10, "BAD": 0}, {"score": 20, "BAD": 1}, {"score": 30, "BAD": 1}],
    )
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})

    loaded = load_dataset("toy", repo_root=tmp_path)

    assert loaded.metadata["row_count"] == 3
    assert loaded.metadata["feature_count"] == 1
    assert loaded.metadata["class_counts"] == {0: 1, 1: 2}
    assert loaded.metadata["default_rate"] == pytest.approx(2 / 3)


def test_portable_path_resolution_ignores_current_working_directory(tmp_path, monkeypatch):
    write_csv(tmp_path, "data/raw/toy.csv", [{"score": 10, "BAD": 0}, {"score": 20, "BAD": 1}])
    write_registry(tmp_path, {"toy": base_spec("toy", "data/raw/toy.csv")})
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)

    loaded = load_dataset("toy", repo_root=tmp_path)

    assert loaded.metadata["source_path"] == "data/raw/toy.csv"


def test_conflicting_registry_metadata_is_rejected(tmp_path):
    spec = base_spec("toy", "data/raw/toy.csv", identifiers=["BAD"])
    write_registry(tmp_path, {"toy": spec})

    with pytest.raises(RegistryError, match="target column"):
        load_registry(repo_root=tmp_path)
