"""Test MCP settings merging functionality."""

import os
from unittest.mock import patch

import pytest

from openhands.core.config.mcp_config import (
    MCPConfig,
    RemoteMCPServer,
    StdioMCPServer,
)
from openhands.sdk.llm import LLM
from openhands.sdk.settings import AgentSettings
from openhands.storage.data_models.settings import Settings


@pytest.fixture(autouse=True)
def allow_short_context_windows():
    with patch.dict(os.environ, {'ALLOW_SHORT_CONTEXT_WINDOWS': 'true'}, clear=False):
        yield


def _mcp_config(settings: Settings) -> MCPConfig | None:
    mcp = settings.agent_settings.mcp_config
    return mcp if mcp and mcp.mcpServers else None


_DEFAULT_LLM = LLM(model='test-model')


def _settings_with_mcp(mcp_config, llm=None):
    """Helper: create Settings with mcp_config set via agent_settings."""
    s = Settings(agent_settings=AgentSettings(llm=llm or _DEFAULT_LLM))
    s.agent_settings.mcp_config = mcp_config
    return s


@pytest.mark.asyncio
async def test_mcp_settings_merge_config_only():
    """Test merging when only config.toml has MCP settings."""
    mock_config_settings = _settings_with_mcp(
        MCPConfig(
            mcpServers={
                'config': RemoteMCPServer(
                    url='http://config-server.com', transport='sse'
                )
            }
        )
    )

    frontend_settings = Settings(agent_settings=AgentSettings(llm=LLM(model='gpt-4')))

    with patch(
        'openhands.storage.data_models.settings.Settings.from_config',
        return_value=mock_config_settings,
    ):
        merged_settings = frontend_settings.merge_with_config_settings()

    merged_mcp_config = _mcp_config(merged_settings)
    assert merged_mcp_config is not None
    assert len(merged_mcp_config.mcpServers) == 1
    assert 'config' in merged_mcp_config.mcpServers
    assert merged_settings.agent_settings.llm.model == 'gpt-4'


@pytest.mark.asyncio
async def test_mcp_settings_merge_frontend_only():
    """Test merging when only frontend has MCP settings."""
    mock_config_settings = Settings(
        agent_settings=AgentSettings(llm=LLM(model='claude-3'))
    )

    frontend_settings = _settings_with_mcp(
        MCPConfig(
            mcpServers={
                'frontend': RemoteMCPServer(
                    url='http://frontend-server.com', transport='sse'
                )
            }
        ),
        llm=LLM(model='gpt-4'),
    )

    with patch(
        'openhands.storage.data_models.settings.Settings.from_config',
        return_value=mock_config_settings,
    ):
        merged_settings = frontend_settings.merge_with_config_settings()

    merged_mcp_config = _mcp_config(merged_settings)
    assert merged_mcp_config is not None
    assert len(merged_mcp_config.mcpServers) == 1
    assert 'frontend' in merged_mcp_config.mcpServers
    assert merged_settings.agent_settings.llm.model == 'gpt-4'


@pytest.mark.asyncio
async def test_mcp_settings_merge_both_present():
    """Test merging when both config.toml and frontend have MCP settings."""
    mock_config_settings = _settings_with_mcp(
        MCPConfig(
            mcpServers={
                'config-sse': RemoteMCPServer(
                    url='http://config-server.com', transport='sse'
                ),
                'config-stdio': StdioMCPServer(command='config-cmd', args=['arg1']),
            }
        )
    )

    frontend_settings = _settings_with_mcp(
        MCPConfig(
            mcpServers={
                'frontend-sse': RemoteMCPServer(
                    url='http://frontend-server.com', transport='sse'
                ),
                'frontend-stdio': StdioMCPServer(command='frontend-cmd', args=['arg2']),
            }
        ),
        llm=LLM(model='gpt-4'),
    )

    with patch(
        'openhands.storage.data_models.settings.Settings.from_config',
        return_value=mock_config_settings,
    ):
        merged_settings = frontend_settings.merge_with_config_settings()

    merged_mcp_config = _mcp_config(merged_settings)
    assert merged_mcp_config is not None
    assert len(merged_mcp_config.mcpServers) == 4
    assert 'config-sse' in merged_mcp_config.mcpServers
    assert 'frontend-sse' in merged_mcp_config.mcpServers
    assert 'config-stdio' in merged_mcp_config.mcpServers
    assert 'frontend-stdio' in merged_mcp_config.mcpServers
    assert merged_settings.agent_settings.llm.model == 'gpt-4'


@pytest.mark.asyncio
async def test_mcp_settings_merge_no_config():
    """Test merging when config.toml has no MCP settings."""
    mock_config_settings = None

    frontend_settings = _settings_with_mcp(
        MCPConfig(
            mcpServers={
                'frontend': RemoteMCPServer(
                    url='http://frontend-server.com', transport='sse'
                )
            }
        ),
        llm=LLM(model='gpt-4'),
    )

    with patch(
        'openhands.storage.data_models.settings.Settings.from_config',
        return_value=mock_config_settings,
    ):
        merged_settings = frontend_settings.merge_with_config_settings()

    merged_mcp_config = _mcp_config(merged_settings)
    assert merged_mcp_config is not None
    assert len(merged_mcp_config.mcpServers) == 1
    assert merged_settings.agent_settings.llm.model == 'gpt-4'


@pytest.mark.asyncio
async def test_mcp_settings_merge_neither_present():
    """Test merging when neither config.toml nor frontend have MCP settings."""
    mock_config_settings = Settings(
        agent_settings=AgentSettings(llm=LLM(model='claude-3'))
    )

    frontend_settings = Settings(agent_settings=AgentSettings(llm=LLM(model='gpt-4')))

    with patch(
        'openhands.storage.data_models.settings.Settings.from_config',
        return_value=mock_config_settings,
    ):
        merged_settings = frontend_settings.merge_with_config_settings()

    assert _mcp_config(merged_settings) is None
    assert merged_settings.agent_settings.llm.model == 'gpt-4'
