"""Compatibility boundary for SDK MCP settings.

OpenHands stores MCP servers through the SDK settings DataModel. SDK 1.30 used
FastMCP's ``{"mcpServers": ...}`` wrapper; software-agent-sdk#3964 makes the
DataModel field the native ``dict[str, MCPServer]`` server map. Keep that shape
decision out of app-server business logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

from fastmcp.mcp_config import MCPConfig as FastMCPConfig
from pydantic import BaseModel

from openhands.sdk.utils.pydantic_secrets import REDACTED_SECRET_VALUE

NativeMCPServer: Any

dump_native_mcp_config: Any

try:
    from openhands.sdk.mcp.config import (
        MCPServer as _NativeMCPServer,
    )
    from openhands.sdk.mcp.config import (
        dump_mcp_config as _dump_native_mcp_config,
    )

    NativeMCPServer = _NativeMCPServer
    dump_native_mcp_config = _dump_native_mcp_config
except ImportError:  # SDK < software-agent-sdk#3964
    NativeMCPServer = None
    dump_native_mcp_config = None


def native_mcp_config_supported() -> bool:
    return NativeMCPServer is not None


def mcp_config_server_map(value: Any) -> dict[str, Any]:
    """Return a name-keyed MCP server map from either SDK settings shape."""
    if not value:
        return {}

    servers: Any
    if isinstance(value, FastMCPConfig):
        servers = value.mcpServers
    elif isinstance(value, Mapping):
        servers = value.get('mcpServers') if 'mcpServers' in value else value
    else:
        servers = getattr(value, 'mcpServers', None)

    return dict(servers) if isinstance(servers, Mapping) else {}


def _dump_server(server: Any) -> Any:
    if isinstance(server, BaseModel):
        return server.model_dump(mode='json', exclude_none=True, exclude_defaults=True)
    return server


def _dump_server_with_secrets(server: Any) -> dict[str, Any]:
    if isinstance(server, BaseModel):
        dumped = server.model_dump(
            mode='json',
            context={'expose_secrets': True},
            exclude_none=True,
            exclude_defaults=True,
        )
    else:
        dumped = deepcopy(server)
    return dumped if isinstance(dumped, dict) else {}


def _is_redacted_secret(value: Any) -> bool:
    return isinstance(value, str) and value in {
        REDACTED_SECRET_VALUE,
        f'Bearer {REDACTED_SECRET_VALUE}',
    }


def _preserve_redacted_leaves(incoming: Any, existing: Any) -> None:
    if not isinstance(incoming, dict) or not isinstance(existing, dict):
        return
    for key, value in list(incoming.items()):
        existing_value = existing.get(key)
        if _is_redacted_secret(value) and existing_value is not None:
            incoming[key] = deepcopy(existing_value)
        elif isinstance(value, dict):
            _preserve_redacted_leaves(value, existing_value)


def _preserve_auth(incoming: dict[str, Any], existing: dict[str, Any]) -> None:
    incoming_auth = incoming.get('auth')
    existing_auth = existing.get('auth')

    if isinstance(incoming_auth, dict) and isinstance(existing_auth, dict):
        same_strategy = incoming_auth.get('strategy') == existing_auth.get('strategy')
        if same_strategy:
            for key, value in existing_auth.items():
                if key not in incoming_auth:
                    incoming_auth[key] = deepcopy(value)
            _preserve_redacted_leaves(incoming_auth, existing_auth)
    elif 'auth' not in incoming and 'headers' not in incoming and existing_auth:
        incoming['auth'] = deepcopy(existing_auth)

    incoming_headers = incoming.get('headers')
    existing_headers = existing.get('headers')
    if isinstance(incoming_headers, dict) and isinstance(existing_headers, dict):
        _preserve_redacted_leaves(incoming_headers, existing_headers)
    elif 'auth' not in incoming and 'headers' not in incoming and existing_headers:
        incoming['headers'] = deepcopy(existing_headers)


def preserve_existing_mcp_secrets(value: Any, existing_value: Any) -> Any:
    """Preserve unchanged MCP secret leaves across redacted settings updates.

    ``mcp_config`` is replaced wholesale when the UI saves settings. The GET
    response cannot expose secret values, so follow-up saves can contain
    ``**********`` placeholders or omit auth details for servers the user did
    not change. For retained server names, carry forward the previous secret
    leaves before validating the replacement payload.
    """
    if value is None or not existing_value:
        return value

    incoming_servers = {
        name: _dump_server_with_secrets(server)
        for name, server in mcp_config_server_map(value).items()
    }
    existing_servers = {
        name: _dump_server_with_secrets(server)
        for name, server in mcp_config_server_map(existing_value).items()
    }

    for name, incoming in incoming_servers.items():
        existing = existing_servers.get(name)
        if not existing:
            continue
        _preserve_auth(incoming, existing)
        incoming_env = incoming.get('env')
        existing_env = existing.get('env')
        if isinstance(incoming_env, dict) and isinstance(existing_env, dict):
            _preserve_redacted_leaves(incoming_env, existing_env)

    return incoming_servers


def normalize_mcp_config_payload(value: Any) -> Any:
    """Normalize an incoming settings payload for the installed SDK version."""
    if value is None:
        return None
    if native_mcp_config_supported():
        return {
            name: _dump_server(server)
            for name, server in mcp_config_server_map(value).items()
        }
    if isinstance(value, FastMCPConfig):
        return value.model_dump(mode='json', exclude_none=True, exclude_defaults=True)
    if isinstance(value, Mapping) and 'mcpServers' not in value:
        return {'mcpServers': dict(value)}
    return value


def replace_mcp_config_in_agent_settings_dump(
    agent_settings_dump: dict[str, Any], value: Any
) -> None:
    """Replace ``mcp_config`` in a dumped agent-settings object."""
    if value is None and native_mcp_config_supported():
        agent_settings_dump.pop('mcp_config', None)
        return
    agent_settings_dump['mcp_config'] = normalize_mcp_config_payload(value)


def make_remote_mcp_server(url: str, headers: dict[str, str]) -> Any:
    """Create a remote MCP server value for the installed SDK settings model."""
    if NativeMCPServer is not None:
        return NativeMCPServer(url=url, headers=headers)
    return {'url': url, 'headers': headers}


def settings_mcp_config_value(mcp_servers: Mapping[str, Any] | None) -> Any:
    """Value suitable for ``OpenHandsAgentSettings.mcp_config``."""
    if not mcp_servers:
        return {} if native_mcp_config_supported() else None
    if native_mcp_config_supported():
        return dict(mcp_servers)
    return FastMCPConfig(
        mcpServers={name: _dump_server(server) for name, server in mcp_servers.items()}
    )


def dump_mcp_config_for_log(mcp_servers: Mapping[str, Any]) -> dict[str, Any]:
    if native_mcp_config_supported() and dump_native_mcp_config is not None:
        try:
            return dump_native_mcp_config(
                cast(Any, mcp_servers),
                context={'expose_secrets': True},
            )
        except Exception:
            pass
    return {name: _dump_server(server) for name, server in mcp_servers.items()}
