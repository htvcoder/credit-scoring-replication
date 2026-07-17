"""Verify Phase 0 credit-scoring datasets from the central registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "datasets.yaml"
CHECKSUM_PATH = ROOT / "data" / "checksums-sha256.csv"

TH02_SCHEMA = [
    "DOB",
    "NKID",
    "DEP",
    "PHON",
    "SINC",
    "AES",
    "DAINC",
    "RES",
    "DHVAL",
    "DMORT",
    "DOUTM",
    "DOUTL",
    "DOUTHP",
    "DOUTCC",
    "BAD",
]


class VerificationError(ValueError):
    """Raised when a configuration or checksum path is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def is_windows_absolute(path_text: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_text)) or path_text.startswith("\\\\")


def validate_relative_path(path_text: str) -> None:
    if not path_text:
        raise VerificationError("Path must not be empty.")
    if "\\" in path_text:
        raise VerificationError(f"Path must use '/' separators, got: {path_text}")
    if is_windows_absolute(path_text):
        raise VerificationError(f"Windows absolute path is not portable: {path_text}")
    path = Path(path_text)
    if path.is_absolute() or path_text.startswith("/"):
        raise VerificationError(f"POSIX absolute path is not portable: {path_text}")
    if ".." in path.parts:
        raise VerificationError(f"Path must not contain '..': {path_text}")


def resolve_repo_path(path_text: str, root: Path = ROOT) -> Path:
    validate_relative_path(path_text)
    return root / Path(path_text)


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    datasets = data.get("datasets", {}) if isinstance(data, dict) else {}
    if not datasets:
        raise VerificationError("Dataset registry is empty or malformed.")
    return datasets


def load_checksums(path: Path = CHECKSUM_PATH) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Path", "Algorithm", "Hash"}
        if set(reader.fieldnames or []) != required:
            raise VerificationError(f"Checksum CSV must have header {sorted(required)}.")
        for row in reader:
            path_text = row["Path"]
            validate_relative_path(path_text)
            if row["Algorithm"].upper() != "SHA256":
                raise VerificationError(f"Unsupported algorithm for {path_text}: {row['Algorithm']}")
            checksums[path_text] = row["Hash"].upper()
    return checksums


def verify_all_checksums(checksums: dict[str, str] | None = None) -> dict[str, Any]:
    checksums = checksums or load_checksums()
    files: dict[str, Any] = {}
    all_pass = True
    for rel_path, expected in checksums.items():
        abs_path = resolve_repo_path(rel_path)
        exists = abs_path.exists()
        actual = sha256_file(abs_path) if exists else None
        passed = exists and actual == expected
        all_pass = all_pass and passed
        files[rel_path] = {
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "pass": passed,
        }
    return {"checksums_pass": all_pass, "files": files}


def active_file_for_spec(spec: dict[str, Any]) -> str:
    if spec.get("processed_file") and spec.get("reader", {}).get("verify_file") == "processed":
        return spec["processed_file"]
    return spec["raw_file"]


def read_dataset(spec: dict[str, Any], path_override: Path | None = None) -> pd.DataFrame:
    reader = spec.get("reader", {})
    path = path_override or resolve_repo_path(active_file_for_spec(spec))
    reader_type = reader.get("type")

    if reader_type == "csv":
        return pd.read_csv(path)
    if reader_type == "excel":
        return pd.read_excel(path, engine=reader.get("engine", "xlrd"), header=reader.get("header", 0))
    if reader_type == "delimited":
        sep = r"\s+" if reader.get("sep") == "whitespace" else reader.get("sep", ",")
        header = None if reader.get("header") is False else "infer"
        return pd.read_csv(path, sep=sep, header=header, names=reader.get("columns"))
    raise VerificationError(f"Unsupported reader type for {spec['id']}: {reader_type}")


def mapped_target(df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    target_cfg = spec.get("target", {})
    target = target_cfg.get("column")
    mapping = target_cfg.get("mapping_to_binary") or {}
    if not target:
        raise VerificationError(f"{spec['id']} has no target column configured.")
    if target not in df.columns:
        raise VerificationError(f"{spec['id']} target column missing from data: {target}")
    if not mapping:
        raise VerificationError(f"{spec['id']} target mapping is not defined.")
    normalized_mapping = {str(key): int(value) for key, value in mapping.items()}
    raw = df[target]
    mapped = raw.map(lambda value: normalized_mapping.get(str(int(value)) if pd.notna(value) and float(value).is_integer() else str(value)))
    return mapped


def profile_dataframe(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    target = spec["target"]["column"]
    numeric = list(spec.get("numeric_columns", []))
    categorical = list(spec.get("categorical_columns", []))
    identifiers = list(spec.get("identifier_columns", []))
    ignored = list(spec.get("ignored_columns", []))
    feature_columns = [
        column
        for column in list(df.columns)
        if column != target and column not in identifiers and column not in ignored
    ]
    mapped = mapped_target(df, spec)
    counts = mapped.value_counts(dropna=False).sort_index()
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "target": target if target in df.columns else None,
        "target_domain_raw": sorted([str(value) for value in df[target].dropna().unique()]) if target in df.columns else [],
        "target_domain_mapped": sorted([int(value) for value in mapped.dropna().unique()]),
        "class_counts": {str(int(key)): int(value) for key, value in counts.items() if pd.notna(key)},
        "default_rate": float((mapped == 1).sum() / len(mapped)) if len(mapped) else None,
        "missing_values": {column: int(df[column].isna().sum()) for column in df.columns},
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "identifier_columns": identifiers,
        "ignored_columns": ignored,
        "feature_columns": feature_columns,
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def validate_registry_spec(spec: dict[str, Any]) -> dict[str, bool]:
    required = [
        "id",
        "full_name",
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
    ]
    checks = {f"registry_has_{field}": field in spec for field in required}
    for path_key in ("raw_file", "processed_file"):
        if spec.get(path_key):
            try:
                validate_relative_path(spec[path_key])
                checks[f"{path_key}_portable"] = True
            except VerificationError:
                checks[f"{path_key}_portable"] = False
    for path_text in spec.get("dictionary_files") or []:
        try:
            validate_relative_path(path_text)
            checks[f"dictionary_path_portable_{path_text}"] = True
        except VerificationError:
            checks[f"dictionary_path_portable_{path_text}"] = False
    return checks


def validate_dataframe(df: pd.DataFrame, spec: dict[str, Any], actual_sha256: str | None = None) -> dict[str, Any]:
    expected = spec["expected"]
    profile = profile_dataframe(df, spec)
    target = spec["target"]["column"]
    numeric = set(spec.get("numeric_columns", []))
    categorical = set(spec.get("categorical_columns", []))
    identifiers = set(spec.get("identifier_columns", []))
    ignored = set(spec.get("ignored_columns", []))
    features = set(profile["feature_columns"])

    checks = validate_registry_spec(spec)
    checks.update(
        {
            "file_exists": True,
            "shape": profile["shape"] == [expected["rows"], expected["total_columns"]],
            "input_count": len(profile["feature_columns"]) == expected["input_count"],
            "target_present": profile["target"] == target,
            "target_mapping_defined": bool(spec.get("target", {}).get("mapping_to_binary")),
            "target_domain": set(profile["target_domain_mapped"]) == {0, 1},
            "class_counts": profile["class_counts"]
            == {
                "0": int(expected["non_default_count"]),
                "1": int(expected["default_count"]),
            },
            "default_rate": abs(profile["default_rate"] - float(expected["default_rate"])) < 1e-9,
            "duplicates": profile["duplicate_rows"] == int(expected.get("duplicate_rows", profile["duplicate_rows"])),
            "metadata_no_overlap_numeric_categorical": numeric.isdisjoint(categorical),
            "target_not_in_features": target not in features,
            "identifier_not_in_features": identifiers.isdisjoint(features),
            "ignored_not_in_features": ignored.isdisjoint(features),
            "metadata_columns_exist": (numeric | categorical | identifiers | ignored | {target}).issubset(set(df.columns)),
            "feature_metadata_complete": features == (numeric | categorical),
        }
    )
    return {
        "dataset": spec["id"],
        "path": active_file_for_spec(spec),
        "sha256": actual_sha256,
        "checks": checks,
        "pass": all(checks.values()),
        "profile": profile,
    }


def verify_dataset(
    dataset_id: str,
    registry: dict[str, dict[str, Any]] | None = None,
    path: Path | None = None,
    checksums: dict[str, str] | None = None,
) -> dict[str, Any]:
    registry = registry or load_registry()
    checksums = checksums or load_checksums()
    if dataset_id not in registry:
        raise VerificationError(f"Unsupported dataset: {dataset_id}")
    spec = registry[dataset_id]
    active_rel = active_file_for_spec(spec)
    active_path = path or resolve_repo_path(active_rel)
    if not active_path.exists():
        return {
            "dataset": dataset_id,
            "path": str(active_path),
            "checks": {"file_exists": False},
            "pass": False,
        }
    actual_sha256 = sha256_file(active_path)
    result = validate_dataframe(read_dataset(spec, path_override=active_path), spec, actual_sha256)
    expected_sha = checksums.get(active_rel)
    result["checks"]["checksum_listed"] = expected_sha is not None
    result["checks"]["checksum_match"] = expected_sha == actual_sha256 if expected_sha else False
    result["pass"] = all(result["checks"].values())
    return result


def main() -> int:
    registry = load_registry()
    dataset_choices = sorted(registry) + ["all"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices, required=True)
    parser.add_argument("--path", type=Path, help="Optional override path for a single dataset.")
    parser.add_argument("--checksums-only", action="store_true", help="Only verify checksum CSV paths and hashes.")
    args = parser.parse_args()

    checksums = load_checksums()
    checksum_result = verify_all_checksums(checksums)
    if args.checksums_only:
        print(json.dumps(checksum_result, indent=2, ensure_ascii=False))
        return 0 if checksum_result["checksums_pass"] else 1

    datasets = sorted(registry) if args.dataset == "all" else [args.dataset]
    if args.path and len(datasets) > 1:
        parser.error("--path can only be used with one dataset")
    results = [
        verify_dataset(dataset, registry=registry, path=args.path, checksums=checksums)
        for dataset in datasets
    ]
    payload: Any = results if len(results) > 1 else results[0]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all(result.get("pass") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
