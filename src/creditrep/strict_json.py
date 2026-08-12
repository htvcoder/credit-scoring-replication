"""Strict JSON object loading for security-sensitive artifact boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StrictJSONError(ValueError):
    """Expected, stable failure while reading an external JSON artifact."""


def _reject_constant(_value: str) -> None:
    raise StrictJSONError("invalid_json_number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise StrictJSONError("duplicate_json_key")
        value[key] = item
    return value


def loads_strict_object(payload: str) -> dict[str, Any]:
    """Parse one JSON object, rejecting ambiguous and non-finite input."""
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except StrictJSONError:
        raise
    except json.JSONDecodeError as exc:
        raise StrictJSONError("malformed_json") from exc
    if not isinstance(value, dict):
        raise StrictJSONError("invalid_json_object")
    return value


def load_strict_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise StrictJSONError("missing_required_artifact") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise StrictJSONError("unreadable_json_artifact") from exc
    return loads_strict_object(payload)
