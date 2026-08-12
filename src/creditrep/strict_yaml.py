"""Strict YAML loading for locked runtime inputs."""

from __future__ import annotations

from typing import Any

import yaml
from yaml.nodes import MappingNode


class StrictYAMLError(yaml.YAMLError):
    """Raised when YAML is ambiguous before application schema validation."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate keys at every mapping level."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise StrictYAMLError("yaml_mapping_node_expected")
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise StrictYAMLError("yaml_mapping_key_unhashable") from exc
            if duplicate:
                raise StrictYAMLError(f"duplicate_yaml_mapping_key:{key!r}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_strict_yaml(stream: Any) -> Any:
    """Safely load YAML while rejecting duplicate mapping keys."""

    return yaml.load(stream, Loader=_UniqueKeySafeLoader)
