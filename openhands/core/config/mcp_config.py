"""MCP configuration — thin wrappers around the SDK MCPConfig from *fastmcp*.

All server configuration uses the unified ``MCPConfig.mcpServers`` dict.
Legacy helpers (``from_toml_section``, ``merge``) are provided for
config.toml parsing and server merging.
"""

from __future__ import annotations

import os
import shlex
from typing import TYPE_CHECKING, Any

from fastmcp.mcp_config import MCPConfig, RemoteMCPServer, StdioMCPServer

if TYPE_CHECKING:
    from openhands.core.config.openhands_config import OpenHandsConfig

from openhands.core.logger import openhands_logger as logger
from openhands.utils.import_utils import get_impl

__all__ = [
    'MCPConfig',
    'RemoteMCPServer',
    'StdioMCPServer',
    'OpenHandsMCPConfig',
    'OpenHandsMCPConfigImpl',
    'merge_mcp_configs',
    'mcp_config_from_toml',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def merge_mcp_configs(base: MCPConfig, other: MCPConfig) -> MCPConfig:
    """Return a new MCPConfig with servers from both configs merged."""
    merged = dict(base.mcpServers)
    merged.update(other.mcpServers)
    return MCPConfig(mcpServers=merged)


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


# ---------------------------------------------------------------------------
# OpenHands default MCP server factory
# ---------------------------------------------------------------------------


class OpenHandsMCPConfig:
    """Factory for the default OpenHands MCP server entries."""

    @staticmethod
    def add_search_engine(
        app_config: 'OpenHandsConfig',
    ) -> dict[str, StdioMCPServer] | None:
        """Return a tavily stdio server entry if a Tavily API key is configured."""
        if (
            app_config.search_api_key
            and app_config.search_api_key.get_secret_value().startswith('tvly-')
        ):
            logger.info('Adding search engine to MCP config')
            return {
                'tavily': StdioMCPServer(
                    command='npx',
                    args=['-y', 'tavily-mcp@0.2.1'],
                    env={
                        'TAVILY_API_KEY': app_config.search_api_key.get_secret_value()
                    },
                )
            }
        logger.warning('No search engine API key found, skipping search engine')
        return None

    @staticmethod
    async def create_default_mcp_server_config(
        host: str, config: 'OpenHandsConfig', user_id: str | None = None
    ) -> dict[str, RemoteMCPServer | StdioMCPServer]:
        """Return a dict of default MCP server entries to merge into config.mcp.

        Returns:
            dict mapping server names to their configs.
        """
        servers: dict[str, RemoteMCPServer | StdioMCPServer] = {}

        search = OpenHandsMCPConfig.add_search_engine(config)
        if search:
            servers.update(search)

        servers['openhands'] = RemoteMCPServer(
            url=f'http://{host}/mcp/mcp',
            transport='http',
            timeout=60,
        )

        return servers


openhands_mcp_config_cls = os.environ.get(
    'OPENHANDS_MCP_CONFIG_CLS',
    'openhands.core.config.mcp_config.OpenHandsMCPConfig',
)

OpenHandsMCPConfigImpl = get_impl(OpenHandsMCPConfig, openhands_mcp_config_cls)
