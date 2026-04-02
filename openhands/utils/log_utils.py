"""Utilities for safe logging, including credential redaction."""

from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

_REDACTED = '***'

# Header names (case-insensitive) that contain credentials
_SENSITIVE_HEADERS = frozenset({
    'authorization',
    'x-session-api-key',
    'x-api-key',
    'api-key',
})

# URL query parameter names (case-insensitive) that contain credentials
_SENSITIVE_QUERY_PARAMS = frozenset({
    'apikey',
    'tavilyapikey',
    'api_key',
    'token',
    'access_token',
    'secret',
    'key',
})

# Dict keys that may hold credential values
_SENSITIVE_KEYS = frozenset({
    'api_key',
    'apiKey',
    'api-key',
    'secret',
    'password',
    'token',
    'access_token',
})

# Environment variable names that commonly hold secrets
_SENSITIVE_ENV_VARS = frozenset({
    'TAVILY_API_KEY',
    'API_KEY',
    'SECRET_KEY',
    'ACCESS_TOKEN',
    'AUTH_TOKEN',
})


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
                    redacted_parts.append(
                        f'{param_name}={v}'
                    )
        redacted_query = '&'.join(redacted_parts)
        return urlunparse(parsed._replace(query=redacted_query))
    except Exception:
        # If URL parsing fails, redact any query string entirely
        return re.sub(r'\?.*', '?***', url)


def _redact_headers(headers: dict[str, Any]) -> dict[str, Any]:
    """Redact values for sensitive header keys."""
    redacted = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADERS:
            redacted[key] = _REDACTED
        else:
            redacted[key] = value
    return redacted


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    """Redact values for sensitive environment variable keys."""
    redacted = {}
    for key, value in env.items():
        if key.upper() in _SENSITIVE_ENV_VARS or any(
            s in key.upper() for s in ('API_KEY', 'SECRET', 'TOKEN', 'PASSWORD')
        ):
            redacted[key] = _REDACTED
        else:
            redacted[key] = value
    return redacted


def redact_mcp_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of an MCP config dict with credentials redacted.

    Handles the V1 dict format: ``{'mcpServers': {'name': {'url': ..., 'headers': ...}}}``

    Redacts:
    - Sensitive header values (Authorization, X-Session-API-Key, etc.)
    - Credential-bearing URL query parameters (tavilyApiKey, api_key, token, etc.)
    - Known sensitive dict keys (api_key, secret, password, token, etc.)
    - Sensitive environment variables (TAVILY_API_KEY, etc.)
    """
    config = copy.deepcopy(config)
    mcp_servers = config.get('mcpServers', {})
    for _server_name, server_cfg in mcp_servers.items():
        if not isinstance(server_cfg, dict):
            continue

        # Redact URL query params
        if 'url' in server_cfg and isinstance(server_cfg['url'], str):
            server_cfg['url'] = _redact_url(server_cfg['url'])

        # Redact headers
        if 'headers' in server_cfg and isinstance(server_cfg['headers'], dict):
            server_cfg['headers'] = _redact_headers(server_cfg['headers'])

        # Redact env vars (e.g. stdio servers)
        if 'env' in server_cfg and isinstance(server_cfg['env'], dict):
            server_cfg['env'] = _redact_env(server_cfg['env'])

        # Redact any top-level sensitive keys in the server config
        for key in list(server_cfg.keys()):
            if key.lower() in _SENSITIVE_KEYS:
                server_cfg[key] = _REDACTED

    return config


def redact_mcp_config_model(config: Any) -> str:
    """Return a safe string representation of an MCPConfig pydantic model.

    Handles the V0 model format with ``.sse_servers``, ``.shttp_servers``, ``.stdio_servers``.

    Redacts API keys, auth headers, and sensitive env vars from the string output.
    """
    text = str(config)
    # Redact api_key='...' patterns
    text = re.sub(
        r"""(api_key=)(['"])[^'"]*\2""",
        r"\1'\2***\2'",
        text,
    )
    # Simpler approach: redact api_key='...' with single or double quotes
    text = re.sub(r"api_key='[^']*'", "api_key='***'", text)
    text = re.sub(r'api_key="[^"]*"', 'api_key="***"', text)

    # Redact sensitive env var values in dict representation
    # Pattern: 'TAVILY_API_KEY': 'value'
    for env_var in _SENSITIVE_ENV_VARS:
        text = re.sub(
            rf"('{env_var}':\s*')[^']*(')",
            rf"\g<1>***\2",
            text,
        )
        text = re.sub(
            rf'("{env_var}":\s*")[^"]*(")',
            rf'\g<1>***\2',
            text,
        )

    # Redact URLs with sensitive query params
    text = re.sub(
        r'((?:tavilyApiKey|apiKey|api_key|token|access_token|secret|key)=)[^&\s\'")\]]+',
        r'\g<1>***',
        text,
        flags=re.IGNORECASE,
    )

    # Redact Authorization header values
    text = re.sub(
        r"('Authorization':\s*')[^']*(')",
        r"\g<1>***\2",
        text,
    )

    # Redact X-Session-API-Key header values
    text = re.sub(
        r"('X-Session-API-Key':\s*')[^']*(')",
        r"\g<1>***\2",
        text,
    )

    return text
