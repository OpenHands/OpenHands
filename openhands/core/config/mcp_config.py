# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from __future__ import annotations

import os
import re
import shlex
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from openhands.core.config.openhands_config import OpenHandsConfig

from openhands.core.logger import openhands_logger as logger
from openhands.utils.import_utils import get_impl


def _validate_mcp_url(url: str) -> str:
    """Shared URL validation logic for MCP servers."""
    if not url.strip():
        raise ValueError('URL cannot be empty')

    url = url.strip()
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError('URL must include a scheme (http:// or https://)')
        if not parsed.netloc:
            raise ValueError('URL must include a valid domain/host')
        if parsed.scheme not in ['http', 'https', 'ws', 'wss']:
            raise ValueError('URL scheme must be http, https, ws, or wss')
        return url
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f'Invalid URL format: {str(e)}')


class MCPSSEServerConfig(BaseModel):
    """Configuration for a single MCP server.

    Attributes:
        url: The server URL
        api_key: Optional API key for authentication
    """

    url: str
    api_key: str | None = None

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format for MCP servers."""
        return _validate_mcp_url(v)


class MCPStdioServerConfig(BaseModel):
    """Configuration for a MCP server that uses stdio.

    Attributes:
        name: The name of the server
        command: The command to run the server
        args: The arguments to pass to the server
        env: The environment variables to set for the server
    """

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator('name', mode='before')
    @classmethod
    def validate_server_name(cls, v: str) -> str:
        """Validate server name for stdio MCP servers."""
        if not v.strip():
            raise ValueError('Server name cannot be empty')

        v = v.strip()

        # Check for valid characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                'Server name can only contain letters, numbers, hyphens, and underscores'
            )

        return v

    @field_validator('command', mode='before')
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command for stdio MCP servers."""
        if not v.strip():
            raise ValueError('Command cannot be empty')

        v = v.strip()

        # Check that command doesn't contain spaces (should be a single executable)
        if ' ' in v:
            raise ValueError(
                'Command should be a single executable without spaces (use arguments field for parameters)'
            )

        return v

    @field_validator('args', mode='before')
    @classmethod
    def parse_args(cls, v) -> list[str]:
        """Parse arguments from string or return list as-is.

        Supports shell-like argument parsing using shlex.split().

        Examples:
        - "-y mcp-remote https://example.com"
        - '--config "path with spaces" --debug'
        - "arg1 arg2 arg3"
        """
        if isinstance(v, str):
            if not v.strip():
                return []

            v = v.strip()

            # Use shell-like parsing for natural argument handling
            try:
                return shlex.split(v)
            except ValueError as e:
                # If shlex parsing fails (e.g., unmatched quotes), provide clear error
                raise ValueError(
                    f'Invalid argument format: {str(e)}. Use shell-like format, e.g., "arg1 arg2" or \'--config "value with spaces"\''
                )

        return v or []

    @field_validator('env', mode='before')
    @classmethod
    def parse_env(cls, v) -> dict[str, str]:
        """Parse environment variables from string or return dict as-is."""
        if isinstance(v, str):
            if not v.strip():
                return {}

            env = {}
            for pair in v.split(','):
                pair = pair.strip()
                if not pair:
                    continue

                if '=' not in pair:
                    raise ValueError(
                        f"Environment variable '{pair}' must be in KEY=VALUE format"
                    )

                key, value = pair.split('=', 1)
                key = key.strip()
                if not key:
                    raise ValueError('Environment variable key cannot be empty')
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                    raise ValueError(
                        f"Invalid environment variable name '{key}'. Must start with letter or underscore, contain only alphanumeric characters and underscores"
                    )

                env[key] = value
            return env
        return v or {}

    def __eq__(self, other):
        """Override equality operator to compare server configurations.

        Two server configurations are considered equal if they have the same
        name, command, args, and env values. The order of args is important,
        but the order of env variables is not.
        """
        if not isinstance(other, MCPStdioServerConfig):
            return False
        return (
            self.name == other.name
            and self.command == other.command
            and self.args == other.args
            and set(self.env.items()) == set(other.env.items())
        )


class MCPSHTTPServerConfig(BaseModel):
    """Configuration for a MCP server that uses SHTTP.

    Attributes:
        url: The server URL
        api_key: Optional API key for authentication
        timeout: Timeout in seconds for tool calls (default: 60s)
    """

    url: str
    api_key: str | None = None
    timeout: int | None = 60

    @field_validator('url', mode='before')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format for MCP servers."""
        return _validate_mcp_url(v)

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v: int | None) -> int | None:
        """Validate timeout value for MCP tool calls."""
        if v is not None:
            if v <= 0:
                raise ValueError('Timeout must be positive')
            if v > 3600:  # 1 hour max
                raise ValueError('Timeout cannot exceed 3600 seconds')
        return v


class MCPConfig(BaseModel):
    """Configuration for MCP (Message Control Protocol) settings.

    Attributes:
        sse_servers: List of MCP SSE server configs
        stdio_servers: List of MCP stdio server configs. These servers will be added to the MCP Router running inside runtime container.
        shttp_servers: List of MCP HTTP server configs.
    """

    sse_servers: list[MCPSSEServerConfig] = Field(default_factory=list)
    stdio_servers: list[MCPStdioServerConfig] = Field(default_factory=list)
    shttp_servers: list[MCPSHTTPServerConfig] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid')

    @staticmethod
    def _normalize_servers(servers_data: list[dict | str]) -> list[dict]:
        """Helper method to normalize SSE server configurations."""
        normalized = []
        for server in servers_data:
            if isinstance(server, str):
                normalized.append({'url': server})
            else:
                normalized.append(server)
        return normalized

    @model_validator(mode='before')
    def convert_string_urls(cls, data):
        """Convert string URLs to MCPSSEServerConfig objects."""
        if isinstance(data, dict):
            if 'sse_servers' in data:
                data['sse_servers'] = cls._normalize_servers(data['sse_servers'])

            if 'shttp_servers' in data:
                data['shttp_servers'] = cls._normalize_servers(data['shttp_servers'])

        return data

    def validate_servers(self) -> None:
        """Validate that server URLs are valid and unique."""
        urls = [server.url for server in self.sse_servers]

        # Check for duplicate server URLs
        if len(set(urls)) != len(urls):
            raise ValueError('Duplicate MCP server URLs are not allowed')

        # Validate URLs
        for url in urls:
            try:
                result = urlparse(url)
                if not all([result.scheme, result.netloc]):
                    raise ValueError(f'Invalid URL format: {url}')
            except Exception as e:
                raise ValueError(f'Invalid URL {url}: {str(e)}')

    @classmethod
    def from_toml_section(cls, data: dict) -> dict[str, 'MCPConfig']:
        """Create a mapping of MCPConfig instances from a toml dictionary representing the [mcp] section.

        The configuration is built from all keys in data.

        Returns:
            dict[str, MCPConfig]: A mapping where the key "mcp" corresponds to the [mcp] configuration
        """
        # Initialize the result mapping
        mcp_mapping: dict[str, MCPConfig] = {}

        try:
            # Convert all entries in sse_servers to MCPSSEServerConfig objects
            if 'sse_servers' in data:
                data['sse_servers'] = cls._normalize_servers(data['sse_servers'])
                servers: list[
                    MCPSSEServerConfig | MCPStdioServerConfig | MCPSHTTPServerConfig
                ] = []
                for server in data['sse_servers']:
                    servers.append(MCPSSEServerConfig(**server))
                data['sse_servers'] = servers

            # Convert all entries in stdio_servers to MCPStdioServerConfig objects
            if 'stdio_servers' in data:
                servers = []
                for server in data['stdio_servers']:
                    servers.append(MCPStdioServerConfig(**server))
                data['stdio_servers'] = servers

            if 'shttp_servers' in data:
                data['shttp_servers'] = cls._normalize_servers(data['shttp_servers'])
                servers = []
                for server in data['shttp_servers']:
                    servers.append(MCPSHTTPServerConfig(**server))
                data['shttp_servers'] = servers

            # Create SSE config if present
            mcp_config = MCPConfig.model_validate(data)
            mcp_config.validate_servers()

            # Create the main MCP config
            mcp_mapping['mcp'] = cls(
                sse_servers=mcp_config.sse_servers,
                stdio_servers=mcp_config.stdio_servers,
                shttp_servers=mcp_config.shttp_servers,
            )
        except ValidationError as e:
            raise ValueError(f'Invalid MCP configuration: {e}')
        return mcp_mapping

    def merge(self, other: MCPConfig):
        return MCPConfig(
            sse_servers=self.sse_servers + other.sse_servers,
            stdio_servers=self.stdio_servers + other.stdio_servers,
            shttp_servers=self.shttp_servers + other.shttp_servers,
        )


class OpenHandsMCPConfig:
    @staticmethod
    def add_search_engine(app_config: 'OpenHandsConfig') -> MCPStdioServerConfig | None:
        """Add search engine to the MCP config."""
        if (
            app_config.search_api_key
            and app_config.search_api_key.get_secret_value().startswith('tvly-')
        ):
            logger.info('Adding search engine to MCP config')
            return MCPStdioServerConfig(
                name='tavily',
                command='npx',
                args=['-y', 'tavily-mcp@0.2.1'],
                env={'TAVILY_API_KEY': app_config.search_api_key.get_secret_value()},
            )
        else:
            logger.warning('No search engine API key found, skipping search engine')
        # Do not add search engine to MCP config in SaaS mode since it will be added by the OpenHands server
        return None

    @staticmethod
    async def create_default_mcp_server_config(
        host: str, config: 'OpenHandsConfig', user_id: str | None = None
    ) -> tuple[MCPSHTTPServerConfig | None, list[MCPStdioServerConfig]]:
        """Create a default MCP server configuration.

        Args:
            host: Host string
            config: OpenHandsConfig
            user_id: Optional user ID for the MCP server
        Returns:
            tuple[MCPSHTTPServerConfig | None, list[MCPStdioServerConfig]]: A tuple containing the default SHTTP server configuration (or None) and a list of MCP stdio server configurations
        """
        stdio_servers = []
        search_engine_stdio_server = OpenHandsMCPConfig.add_search_engine(config)
        if search_engine_stdio_server:
            stdio_servers.append(search_engine_stdio_server)

        shttp_servers = MCPSHTTPServerConfig(url=f'http://{host}/mcp/mcp', api_key=None)

        return shttp_servers, stdio_servers


openhands_mcp_config_cls = os.environ.get(
    'OPENHANDS_MCP_CONFIG_CLS',
    'openhands.core.config.mcp_config.OpenHandsMCPConfig',
)

OpenHandsMCPConfigImpl = get_impl(OpenHandsMCPConfig, openhands_mcp_config_cls)
