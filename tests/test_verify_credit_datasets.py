from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from convert_th02 import convert_th02
from verify_credit_datasets import (
    REGISTRY_PATH,
    TH02_SCHEMA,
    VerificationError,
    load_checksums,
    load_registry,
    resolve_repo_path,
    sha256_file,
    validate_dataframe,
    verify_all_checksums,
    verify_dataset,
)


def require_path(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Required local data file is missing: {path}")


def test_registry_has_all_required_datasets_and_fields():
    registry = load_registry()
    assert set(registry) == {"ac", "gc", "hmeq", "th02", "tc", "gmc"}
    required = {
        "id",
        "full_name",
        "active_file",
        "raw_file",
        "target",
        "expected",
        "numeric_columns",
        "categorical_columns",
        "identifier_columns",
        "ignored_columns",
        "missing_values",
        "source",
        "license",
        "raw_preprocessing_caveat",
        "usable",
        "deviation_notes",
    }
    for dataset_id, spec in registry.items():
        assert required.issubset(spec), dataset_id
        assert spec["target"].get("mapping_to_binary"), dataset_id


def test_target_mapping_for_all_datasets():
    registry = load_registry()
    assert registry["ac"]["target"]["mapping_to_binary"] == {0: 0, 1: 1}
    assert registry["gc"]["target"]["mapping_to_binary"] == {1: 0, 2: 1}
    assert registry["hmeq"]["target"]["mapping_to_binary"] == {0: 0, 1: 1}
    assert registry["th02"]["target"]["mapping_to_binary"] == {0: 0, 1: 1}
    assert registry["tc"]["target"]["mapping_to_binary"] == {0: 0, 1: 1}
    assert registry["gmc"]["target"]["mapping_to_binary"] == {0: 0, 1: 1}


def test_metadata_feature_sets_are_consistent():
    registry = load_registry()
    for dataset_id, spec in registry.items():
        numeric = set(spec["numeric_columns"])
        categorical = set(spec["categorical_columns"])
        identifiers = set(spec["identifier_columns"])
        ignored = set(spec["ignored_columns"])
        target = spec["target"]["column"]
        assert numeric.isdisjoint(categorical), dataset_id
        assert target not in numeric | categorical | identifiers | ignored, dataset_id
        assert identifiers.isdisjoint(numeric | categorical), dataset_id
        assert ignored.isdisjoint(numeric | categorical), dataset_id


@pytest.mark.parametrize("dataset", ["ac", "gc", "hmeq", "th02", "tc", "gmc"])
@pytest.mark.raw_data
def test_integration_dataset_validation_passes(dataset):
    registry = load_registry()
    active = registry[dataset].get("processed_file") if registry[dataset].get("reader", {}).get("verify_file") == "processed" else registry[dataset]["raw_file"]
    require_path(resolve_repo_path(active))
    result = verify_dataset(dataset, registry=registry)
    assert result["pass"], result


@pytest.mark.raw_data
def test_hmeq_active_file_is_full_and_604_file_fails_core_validation():
    registry = load_registry()
    require_path(ROOT / "data" / "raw" / "hmeq" / "hmeq_full.csv")
    full_result = verify_dataset("hmeq", registry=registry)
    assert full_result["profile"]["shape"] == [5960, 13]
    assert full_result["profile"]["class_counts"] == {"0": 4771, "1": 1189}

    old_path = ROOT / "data" / "raw" / "hmeq" / "hmeq.csv"
    require_path(old_path)
    old_result = verify_dataset("hmeq", registry=registry, path=old_path)
    assert not old_result["pass"]
    assert not old_result["checks"]["shape"]
    assert not old_result["checks"]["class_counts"]


@pytest.mark.raw_data
def test_th02_conversion_preserves_shape_class_counts_duplicates_and_raw_hash(tmp_path):
    raw_path = ROOT / "data" / "raw" / "th02" / "public.xls"
    require_path(raw_path)
    before = sha256_file(raw_path)
    out_path = tmp_path / "th02.csv"
    result = convert_th02(raw_path, out_path)
    after = sha256_file(raw_path)
    assert result["method"] in {"xlrd", "libreoffice"}
    assert out_path.exists()
    assert before == after
    df = pd.read_csv(out_path)
    assert list(df.columns) == TH02_SCHEMA
    assert df.shape == (1225, 15)
    assert df["BAD"].value_counts().sort_index().to_dict() == {0: 902, 1: 323}
    assert int(df.duplicated().sum()) == 23


def test_missing_bad_target_fails_validation():
    registry = load_registry()
    spec = dict(registry["hmeq"])
    columns = [column for column in spec["numeric_columns"] + spec["categorical_columns"] if column != "BAD"]
    df = pd.DataFrame([[0] * len(columns)], columns=columns)
    with pytest.raises(VerificationError):
        validate_dataframe(df, spec)


def test_bad_target_domain_fails_validation():
    registry = load_registry()
    spec = registry["hmeq"]
    columns = spec["numeric_columns"] + spec["categorical_columns"] + [spec["target"]["column"]]
    df = pd.DataFrame([[0] * (len(columns) - 1) + [2]], columns=columns)
    result = validate_dataframe(df, spec)
    assert not result["pass"]
    assert not result["checks"]["target_domain"]


def test_checksum_csv_uses_only_relative_portable_paths():
    checksums = load_checksums()
    assert checksums
    assert all(":\\" not in path for path in checksums)
    assert all(not path.startswith("/") for path in checksums)
    assert all("\\" not in path for path in checksums)
    assert all(".." not in Path(path).parts for path in checksums)


BAD_PATHS = [
    f"{'D'}:/data/raw/file.csv",
    f"{'D'}:\\data\\raw\\file.csv",
    "/" + "data/raw/file.csv",
    "data/../raw/file.csv",
]


@pytest.mark.parametrize("bad_path", BAD_PATHS)
def test_checksum_resolver_rejects_non_portable_paths(bad_path):
    with pytest.raises(VerificationError):
        resolve_repo_path(bad_path)


def test_relative_path_resolves_from_repo_root():
    resolved = resolve_repo_path("data/checksums-sha256.csv")
    assert resolved == ROOT / "data" / "checksums-sha256.csv"


@pytest.mark.raw_data
def test_checksum_verification_passes_for_present_files():
    result = verify_all_checksums()
    assert result["checksums_pass"], result


@pytest.mark.raw_data
def test_verifier_runs_from_non_root_cwd(tmp_path):
    script = ROOT / "scripts" / "verify_credit_datasets.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--dataset", "gc"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_dependencies_declared():
    runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    for package in ["pandas==", "numpy==", "xlrd==", "PyYAML=="]:
        assert package in runtime
    assert "pytest==" in dev


def test_sha256_file_is_stable(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_bytes(b"BAD,LOAN\n0,100\n")
    expected = hashlib.sha256(b"BAD,LOAN\n0,100\n").hexdigest().upper()
    assert sha256_file(sample) == expected
