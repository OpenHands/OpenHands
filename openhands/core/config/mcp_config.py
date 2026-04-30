"""MCP configuration — thin wrappers around the SDK MCPConfig from *fastmcp*.

All server configuration uses the unified ``MCPConfig.mcpServers`` dict.
Legacy helpers (``from_toml_section``) are provided for config.toml parsing.
"""

from __future__ import annotations

import shlex
from typing import Any

from fastmcp.mcp_config import MCPConfig, RemoteMCPServer, StdioMCPServer

__all__ = [
    'MCPConfig',
    'RemoteMCPServer',
    'StdioMCPServer',
    'mcp_config_from_toml',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_stdio_args(v: Any) -> list[str]:
    """Parse stdio args from a string using shlex or return list as-is."""
    if isinstance(v, str):
        if not v.strip():
            return []
        return shlex.split(v.strip())
    return list(v or [])


def _parse_stdio_env(v: Any) -> dict[str, str]:
    """Parse stdio env from a comma-separated string or return dict as-is."""
    if isinstance(v, str):
        env: dict[str, str] = {}
        for pair in v.split(','):
            pair = pair.strip()
            if not pair:
                continue
            if '=' not in pair:
                raise ValueError(
                    f"Environment variable '{pair}' must be in KEY=VALUE format"
                )
            key, value = pair.split('=', 1)
            env[key.strip()] = value
        return env
    return dict(v or {})


def mcp_config_from_toml(data: dict[str, Any]) -> dict[str, MCPConfig]:
    """Parse a ``[mcp]`` TOML section into ``{'mcp': MCPConfig}``.

    Accepts the legacy ``sse_servers`` / ``shttp_servers`` / ``stdio_servers``
    list format and converts to the unified ``mcpServers`` dict.
    """
    servers: dict[str, RemoteMCPServer | StdioMCPServer] = {}

    for entry in data.get('sse_servers', []):
        if isinstance(entry, str):
            entry = {'url': entry}
        name = f'sse_{len([k for k in servers if k.startswith("sse_")])}'
        servers[name] = RemoteMCPServer(
            url=entry['url'],
            transport='sse',
            auth=entry.get('api_key'),
        )

    for entry in data.get('shttp_servers', []):
        if isinstance(entry, str):
            entry = {'url': entry}
        name = f'shttp_{len([k for k in servers if k.startswith("shttp_")])}'
        servers[name] = RemoteMCPServer(
            url=entry['url'],
            transport='http',
            auth=entry.get('api_key'),
            timeout=entry.get('timeout', 60),
        )

    for entry in data.get('stdio_servers', []):
        name = entry.get(
            'name', f'stdio_{len([k for k in servers if k.startswith("stdio_")])}'
        )
        servers[name] = StdioMCPServer(
            command=entry['command'],
            args=_parse_stdio_args(entry.get('args', [])),
            env=_parse_stdio_env(entry.get('env', {})),
        )

    return {'mcp': MCPConfig(mcpServers=servers)}
