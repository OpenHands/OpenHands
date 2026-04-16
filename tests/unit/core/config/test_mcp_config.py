import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from openhands.controller.agent import Agent
from openhands.core.config import OpenHandsConfig, load_from_env
from openhands.core.config.mcp_config import (
    MCPConfig,
    RemoteMCPServer,
    StdioMCPServer,
    mcp_config_from_toml,
    merge_mcp_configs,
)
from openhands.llm.llm_registry import LLMRegistry
from openhands.server.services.conversation_stats import ConversationStats
from openhands.server.session.conversation_init_data import ConversationInitData
from openhands.server.session.session import Session
from openhands.storage.memory import InMemoryFileStore


def test_valid_remote_config():
    """Test a valid remote server configuration."""
    config = MCPConfig(
        mcpServers={
            'server1': RemoteMCPServer(url='http://server1:8080', transport='sse'),
            'server2': RemoteMCPServer(url='http://server2:8080', transport='http'),
        }
    )
    assert len(config.mcpServers) == 2


def test_empty_config():
    """Test configuration with no servers."""
    config = MCPConfig(mcpServers={})
    assert len(config.mcpServers) == 0


def test_remote_server_config_with_auth():
    """Test RemoteMCPServer with auth token."""
    config = RemoteMCPServer(
        url='http://server1:8080', transport='sse', auth='test-api-key'
    )
    assert config.url == 'http://server1:8080'
    assert config.auth == 'test-api-key'


def test_remote_server_config_without_auth():
    """Test RemoteMCPServer without auth token."""
    config = RemoteMCPServer(url='http://server1:8080', transport='sse')
    assert config.url == 'http://server1:8080'
    assert config.auth is None


def test_mcp_stdio_server_config_basic():
    """Test basic StdioMCPServer."""
    config = StdioMCPServer(command='python')
    assert config.command == 'python'
    assert config.args == []
    assert config.env == {}


def test_mcp_stdio_server_config_with_args_and_env():
    """Test StdioMCPServer with args and env."""
    config = StdioMCPServer(
        command='python',
        args=['-m', 'server'],
        env={'DEBUG': 'true', 'PORT': '8080'},
    )
    assert config.command == 'python'
    assert config.args == ['-m', 'server']
    assert config.env == {'DEBUG': 'true', 'PORT': '8080'}


def test_mcp_config_with_stdio_servers():
    """Test MCPConfig with stdio servers."""
    config = MCPConfig(
        mcpServers={
            'test-server': StdioMCPServer(
                command='python',
                args=['-m', 'server'],
                env={'DEBUG': 'true'},
            )
        }
    )
    assert len(config.mcpServers) == 1
    server = config.mcpServers['test-server']
    assert isinstance(server, StdioMCPServer)
    assert server.command == 'python'
    assert server.args == ['-m', 'server']
    assert server.env == {'DEBUG': 'true'}


def test_from_toml_section_valid():
    """Test creating config from valid TOML section."""
    data = {
        'sse_servers': [{'url': 'http://server1:8080'}],
    }
    result = mcp_config_from_toml(data)
    assert 'mcp' in result
    mcp = result['mcp']
    assert len(mcp.mcpServers) == 1
    server = list(mcp.mcpServers.values())[0]
    assert isinstance(server, RemoteMCPServer)
    assert server.url == 'http://server1:8080'


def test_from_toml_section_with_stdio_servers():
    """Test creating config from TOML section with stdio servers."""
    data = {
        'sse_servers': [{'url': 'http://server1:8080'}],
        'stdio_servers': [
            {
                'name': 'test-server',
                'command': 'python',
                'args': ['-m', 'server'],
                'env': {'DEBUG': 'true'},
            }
        ],
    }
    result = mcp_config_from_toml(data)
    assert 'mcp' in result
    mcp = result['mcp']
    assert len(mcp.mcpServers) == 2
    assert 'test-server' in mcp.mcpServers
    stdio = mcp.mcpServers['test-server']
    assert isinstance(stdio, StdioMCPServer)
    assert stdio.command == 'python'
    assert stdio.args == ['-m', 'server']
    assert stdio.env == {'DEBUG': 'true'}


def test_mcp_config_with_both_server_types():
    """Test MCPConfig with both remote and stdio servers."""
    config = MCPConfig(
        mcpServers={
            'remote': RemoteMCPServer(
                url='http://server1:8080', transport='sse', auth='test-api-key'
            ),
            'local': StdioMCPServer(
                command='python',
                args=['-m', 'server'],
                env={'DEBUG': 'true'},
            ),
        }
    )
    assert len(config.mcpServers) == 2
    remote = config.mcpServers['remote']
    assert isinstance(remote, RemoteMCPServer)
    assert remote.url == 'http://server1:8080'
    assert remote.auth == 'test-api-key'
    local = config.mcpServers['local']
    assert isinstance(local, StdioMCPServer)
    assert local.command == 'python'


def test_mcp_config_model_validation_error():
    """Test MCPConfig validation error with invalid data."""
    with pytest.raises(ValidationError):
        # Missing required 'url' field
        RemoteMCPServer()

    with pytest.raises(ValidationError):
        # Missing required 'command' field
        StdioMCPServer()


def test_merge_mcp_configs():
    """Test merging two MCPConfig instances."""
    c1 = MCPConfig(
        mcpServers={
            'a': RemoteMCPServer(url='http://a.com', transport='sse'),
        }
    )
    c2 = MCPConfig(
        mcpServers={
            'b': StdioMCPServer(command='echo'),
        }
    )
    merged = merge_mcp_configs(c1, c2)
    assert len(merged.mcpServers) == 2
    assert 'a' in merged.mcpServers
    assert 'b' in merged.mcpServers


def test_merge_mcp_configs_override():
    """Test that merge_mcp_configs uses the 'other' value for duplicate keys."""
    c1 = MCPConfig(
        mcpServers={
            'same': RemoteMCPServer(url='http://old.com', transport='sse'),
        }
    )
    c2 = MCPConfig(
        mcpServers={
            'same': RemoteMCPServer(url='http://new.com', transport='http'),
        }
    )
    merged = merge_mcp_configs(c1, c2)
    assert len(merged.mcpServers) == 1
    assert merged.mcpServers['same'].url == 'http://new.com'


def test_stdio_server_equality():
    """Test StdioMCPServer equality."""
    server1 = StdioMCPServer(
        command='python',
        args=['--verbose', '--debug', '--port=8080'],
        env={'DEBUG': 'true', 'PORT': '8080'},
    )

    server2 = StdioMCPServer(
        command='python',
        args=['--verbose', '--debug', '--port=8080'],
        env={'PORT': '8080', 'DEBUG': 'true'},
    )

    assert server1 == server2

    server3 = StdioMCPServer(
        command='python',
        args=['--debug', '--port=8080', '--verbose'],
        env={'DEBUG': 'true', 'PORT': '8080'},
    )

    assert server1 != server3


def test_toml_mcp_config_loads(tmp_path):
    """Test that TOML [mcp] sections with legacy format load correctly."""
    toml_file = tmp_path / 'config.toml'
    with open(toml_file, 'w', encoding='utf-8') as f:
        f.write("""
[mcp]
sse_servers = [{ url = "http://toml-server:8080" }]
shttp_servers = [
    { url = "http://toml-http-server:8080", api_key = "toml-api-key" }
]
""")

    config = OpenHandsConfig()

    from openhands.core.config import load_from_toml

    load_from_toml(config, str(toml_file))

    # Verify TOML values were loaded as SDK MCPConfig
    assert len(config.mcp.mcpServers) >= 2


@pytest.mark.asyncio
async def test_session_preserves_mcp_config(monkeypatch, tmp_path):
    """Test that Session preserves MCP configuration from TOML."""
    toml_file = tmp_path / 'config.toml'
    with open(toml_file, 'w', encoding='utf-8') as f:
        f.write("""
[mcp]
shttp_servers = [
    { url = "http://test-server:8080", api_key = "test-key" }
]
""")

    monkeypatch.setenv('MCP_HOST', 'dummy')

    config = OpenHandsConfig()

    from openhands.core.config import load_from_toml

    load_from_toml(config, str(toml_file))
    load_from_env(config, os.environ)

    assert config.mcp_host == 'dummy'
    assert len(config.mcp.mcpServers) >= 1

    session = Session(
        sid='test-sid',
        file_store=InMemoryFileStore({}),
        config=config,
        sio=AsyncMock(),
        llm_registry=LLMRegistry(config=OpenHandsConfig()),
        conversation_stats=ConversationStats(None, 'test-sid', None),
    )

    settings = ConversationInitData()

    mock_agent_cls = MagicMock()
    mock_agent_instance = MagicMock()
    mock_agent_cls.return_value = mock_agent_instance

    with (
        patch.object(session.agent_session, 'start', AsyncMock()),
        patch.object(Agent, 'get_cls', return_value=mock_agent_cls),
    ):
        await session.initialize_agent(settings, None, None)

    assert isinstance(session.config.mcp, MCPConfig)

    await session.close()
