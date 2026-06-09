from __future__ import annotations

from typing import Any

# Keys that should be replaced wholesale (not deep merged) because they
# represent sets of items where merging would resurrect deleted entries.
WHOLESALE_REPLACEMENT_KEYS: frozenset[str] = frozenset({'mcp_config'})


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


def deep_merge_with_wholesale_keys(
    base: dict[str, Any],
    updates: dict[str, Any],
    wholesale_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Like deep_merge, but specified keys are replaced wholesale (not merged).

    Useful for keys like 'mcp_config' where the dict represents a set of items
    and deep merging would resurrect deleted items.

    Args:
        base: The base dictionary to merge into.
        updates: The updates to apply.
        wholesale_keys: Keys that should be replaced entirely from updates
            rather than deep merged. Defaults to WHOLESALE_REPLACEMENT_KEYS.

    Returns:
        A new dictionary with updates applied.
    """
    if wholesale_keys is None:
        wholesale_keys = WHOLESALE_REPLACEMENT_KEYS

    result = deep_merge(base, updates)
    for key in wholesale_keys:
        if key in updates:
            result[key] = updates[key]
    return result
