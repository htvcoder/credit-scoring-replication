"""Canonical split hashing."""

from __future__ import annotations

from typing import Any

from creditrep.config.loader import sha256_canonical


def split_hash(payload: dict[str, Any]) -> str:
    return sha256_canonical(payload)
