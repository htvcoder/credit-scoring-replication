"""Convert TH02 public.xls to a validated CSV without modifying raw input."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from verify_credit_datasets import TH02_SCHEMA, load_registry, validate_dataframe


def read_with_xlrd(input_path: Path) -> pd.DataFrame:
    return pd.read_excel(input_path, engine="xlrd")


def read_with_libreoffice(input_path: Path) -> pd.DataFrame:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        raise RuntimeError(
            "LibreOffice/soffice is not installed or not on PATH; install it or use xlrd."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        command = [
            executable,
            "--headless",
            "--convert-to",
            "csv",
            "--outdir",
            str(tmp_path),
            str(input_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "LibreOffice conversion failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        converted = tmp_path / f"{input_path.stem}.csv"
        if not converted.exists():
            raise RuntimeError("LibreOffice did not produce the expected CSV output.")
        return pd.read_csv(converted)


def convert_th02(input_path: Path, output_path: Path) -> dict:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Output path must not be the raw input path.")

    try:
        df = read_with_xlrd(input_path)
        method = "xlrd"
    except Exception as xlrd_error:
        try:
            df = read_with_libreoffice(input_path)
            method = "libreoffice"
        except Exception as libreoffice_error:
            raise RuntimeError(
                f"Could not read TH02 with xlrd ({xlrd_error!r}) or LibreOffice "
                f"({libreoffice_error!r})."
            ) from libreoffice_error

    if list(df.columns) != TH02_SCHEMA:
        df.columns = TH02_SCHEMA

    registry = load_registry()
    validation = validate_dataframe(df, registry["th02"])
    if not validation["pass"]:
        raise ValueError(f"Converted TH02 failed validation: {validation['checks']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, lineterminator="\n")
    return {"method": method, "output": str(output_path), "validation": validation}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = convert_th02(args.input, args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
