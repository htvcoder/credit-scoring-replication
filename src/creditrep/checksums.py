"""Checksum helpers reused by dataset verification and experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from creditrep.datasets.exceptions import DatasetFileError
from creditrep.datasets.registry import find_repo_root, resolve_repo_path, validate_portable_path


@dataclass(frozen=True)
class DatasetChecksum:
    """Declared and actual checksum status for one active dataset file."""

    dataset_id: str
    source_file: str
    declared_sha256: str
    actual_sha256: str
    matches: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_checksum_registry(
    checksum_path: Path | str | None = None,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, str]:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    path = Path(checksum_path) if checksum_path is not None else root / "data" / "checksums-sha256.csv"
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise DatasetFileError(f"Checksum registry does not exist: {path}")
    checksums: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"Path", "Algorithm", "Hash"}:
            raise DatasetFileError(f"Checksum CSV must have Path, Algorithm, Hash columns: {path}")
        for row in reader:
            rel_path = row["Path"]
            validate_portable_path(rel_path, context="checksums-sha256.csv")
            if row["Algorithm"].upper() != "SHA256":
                raise DatasetFileError(f"Unsupported checksum algorithm for {rel_path}: {row['Algorithm']}")
            checksums[rel_path] = row["Hash"].upper()
    return checksums


def get_dataset_checksum(
    dataset_id: str,
    source_file: str,
    *,
    repo_root: Path | str | None = None,
    checksum_path: Path | str | None = None,
) -> DatasetChecksum:
    root = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
    checksums = load_checksum_registry(checksum_path, repo_root=root)
    declared = checksums.get(source_file)
    if declared is None:
        raise DatasetFileError(f"{dataset_id}: active file {source_file} is not listed in checksum registry.")
    source_path = resolve_repo_path(source_file, repo_root=root, context=f"{dataset_id}.source_file")
    if not source_path.exists():
        raise DatasetFileError(f"{dataset_id}: active file does not exist for checksum: {source_file}")
    actual = sha256_file(source_path)
    if actual != declared:
        raise DatasetFileError(
            f"{dataset_id}: checksum mismatch for {source_file}; declared {declared}, actual {actual}."
        )
    return DatasetChecksum(
        dataset_id=dataset_id,
        source_file=source_file,
        declared_sha256=declared,
        actual_sha256=actual,
        matches=True,
    )
