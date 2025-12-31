from __future__ import annotations

from typing import Any


def apply_defaults_if_absent(
    user_kwargs: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    """Return a new dict with defaults applied when keys are absent.

    - Pure and deterministic; does not mutate inputs
    - Only applies defaults when the key is missing and default is not None
    - Does not alter user-provided values
    """
    out = dict(user_kwargs)
    for key, value in defaults.items():
        if key not in out and value is not None:
            out[key] = value
    return out
