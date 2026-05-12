from __future__ import annotations

from typing import Any


def deep_merge(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge *updates* into a shallow copy of *base*.

    * Nested dicts are merged recursively.
    * ``None`` values in *updates* remove the corresponding key.
    * All other values overwrite.
    """
    result: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
