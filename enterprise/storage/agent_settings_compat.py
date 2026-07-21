from typing import Any


def normalize_legacy_empty_tools(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(settings)
    if normalized.get('tools') == []:
        normalized['tools'] = None
    return normalized
