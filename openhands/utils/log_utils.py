"""Utilities for safe logging, including credential redaction.

Delegates core dict redaction to ``openhands.sdk.utils.redact.sanitize_dict``
and adds URL-query-param redaction on top (which the SDK does not yet handle).

Source of truth for sanitize_dict / is_secret_key:
    openhands-sdk/openhands/sdk/utils/redact.py
    in repo: https://github.com/OpenHands/software-agent-sdk
"""

from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

from openhands.sdk.utils.redact import sanitize_dict

_REDACTED = '<redacted>'

# URL query parameter names (case-insensitive) that contain credentials.
# The SDK's sanitize_dict handles dict keys but not URL query params embedded
# in string values, so we keep this layer.
_SENSITIVE_QUERY_PARAMS = frozenset(
    {
        'apikey',
        'tavilyapikey',
        'api_key',
        'token',
        'access_token',
        'secret',
        'key',
    }
)


def _redact_url(url: str) -> str:
    """Redact credential-bearing query parameters from a URL."""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        params = parse_qs(parsed.query, keep_blank_values=True)
        redacted_parts: list[str] = []
        for param_name, values in params.items():
            if param_name.lower() in _SENSITIVE_QUERY_PARAMS:
                redacted_parts.append(f'{param_name}={_REDACTED}')
            else:
                for v in values:
                    redacted_parts.append(f'{param_name}={v}')
        redacted_query = '&'.join(redacted_parts)
        return urlunparse(parsed._replace(query=redacted_query))
    except Exception:
        # If URL parsing fails, redact any query string entirely
        return re.sub(r'\?.*', '?<redacted>', url)


def _walk_redact_urls(obj: Any) -> Any:
    """Walk a nested dict/list and apply ``_redact_url`` to every string value."""
    if isinstance(obj, dict):
        return {k: _walk_redact_urls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_redact_urls(item) for item in obj]
    if isinstance(obj, str) and '?' in obj:
        return _redact_url(obj)
    return obj


def redact_mcp_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of an MCP config dict with credentials redacted.

    Handles the V1 dict format: ``{'mcpServers': {'name': {'url': ..., 'headers': ...}}}``

    Uses the SDK's ``sanitize_dict`` for key-based redaction (headers, env,
    api_key, secret, token, etc.) then applies URL-query-param redaction on
    string values that contain ``?``.
    """
    config = copy.deepcopy(config)
    # SDK handles key-based redaction (headers, env, api_key, token, etc.)
    config = sanitize_dict(config)
    # Walk the result to redact sensitive URL query params in string values
    config = _walk_redact_urls(config)
    return config


def redact_mcp_config_model(config: Any) -> str:
    """Return a safe string representation of an MCPConfig pydantic model.

    Handles the V0 model format with ``.sse_servers``, ``.shttp_servers``,
    ``.stdio_servers``.

    Redacts API keys, auth headers, and sensitive env vars from the string
    output using regex on the ``str(model)`` representation.
    """
    text = str(config)

    # Redact api_key='...' patterns (single or double quotes)
    text = re.sub(r"api_key='[^']*'", "api_key='<redacted>'", text)
    text = re.sub(r'api_key="[^"]*"', 'api_key="<redacted>"', text)

    # Redact sensitive env var values in dict representation.
    # Match any key containing KEY, SECRET, TOKEN, PASSWORD (case-insensitive).
    text = re.sub(
        r"('[A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*':\s*')[^']*(')",
        r'\g<1><redacted>\2',
        text,
    )
    text = re.sub(
        r'("[A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*":\s*")[^"]*(")',
        r'\g<1><redacted>\2',
        text,
    )

    # Redact URLs with sensitive query params
    text = re.sub(
        r'((?:tavilyApiKey|apiKey|api_key|token|access_token|secret|key)=)[^&\s\'")\]]+',
        r'\g<1><redacted>',
        text,
        flags=re.IGNORECASE,
    )

    # Redact Authorization header values
    text = re.sub(
        r"('Authorization':\s*')[^']*(')",
        r'\g<1><redacted>\2',
        text,
    )

    # Redact X-Session-API-Key header values
    text = re.sub(
        r"('X-Session-API-Key':\s*')[^']*(')",
        r'\g<1><redacted>\2',
        text,
    )

    return text
