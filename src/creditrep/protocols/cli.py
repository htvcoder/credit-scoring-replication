"""CLI P7A: chỉ kiểm tra/hiển thị/khóa manifest; không chạy thực nghiệm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .p7a import ProtocolManifestError, load_manifest, manifest_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "verify", "render", "lock"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest, verify_lock=args.command != "lock")
        if args.command == "lock":
            manifest["lock"]["manifest_sha256"] = manifest_hash(manifest)
            args.manifest.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
        if args.command == "render":
            print(json.dumps({"protocol": manifest["protocol"], "datasets": manifest["datasets"], "cross_validation": manifest["cross_validation"]}, ensure_ascii=False, indent=2))
        else:
            print(f"P7A manifest valid: {args.manifest}")
        return 0
    except ProtocolManifestError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
