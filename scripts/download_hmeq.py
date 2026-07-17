"""Download the active HMEQ full CSV artifact without overwriting by default."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from verify_credit_datasets import load_registry, verify_dataset


ROOT = Path(__file__).resolve().parents[1]


def download_hmeq(output: Path | None = None, force: bool = False) -> dict:
    registry = load_registry()
    spec = registry["hmeq"]
    url = spec["source_url"]
    output_path = output or ROOT / spec["raw_file"]
    output_path = output_path.resolve()

    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    output_path.write_bytes(content)

    result = verify_dataset("hmeq", registry=registry, path=output_path)
    if not result["pass"]:
        raise ValueError(f"Downloaded HMEQ failed validation: {result['checks']}")
    return {"output": str(output_path), "sha256": result["sha256"], "validation": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_hmeq(output=args.output, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
