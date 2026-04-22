"""Integration test for MCP settings merging in the full flow."""

from unittest.mock import AsyncMock, patch

import pytest

from openhands.core.config.mcp_config import MCPConfig, RemoteMCPServer
from openhands.sdk.llm import LLM
from openhands.sdk.settings import AgentSettings
from openhands.server.user_auth.default_user_auth import DefaultUserAuth
from openhands.storage.data_models.settings import Settings
from openhands.storage.settings.file_settings_store import FileSettingsStore


def _sdk_mcp_config(settings: Settings) -> MCPConfig | None:
    return settings.agent_settings.mcp_config


@pytest.mark.asyncio
async def test_user_auth_mcp_merging_integration():
    """Test that MCP merging works in the user auth flow."""
    config_settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='config-model'),
            mcp_config=MCPConfig(
                mcpServers={
                    'config': RemoteMCPServer(
                        url='http://config-server.com', transport='sse'
                    )
                }
            ),
        ),
    )

    stored_settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='anthropic/claude-sonnet-4-5-20250929'),
            mcp_config=MCPConfig(
                mcpServers={
                    'frontend': RemoteMCPServer(
                        url='http://frontend-server.com', transport='sse'
                    )
                }
            ),
        ),
    )

    user_auth = DefaultUserAuth()

    mock_settings_store = AsyncMock(spec=FileSettingsStore)
    mock_settings_store.load.return_value = stored_settings

    with patch.object(
        user_auth, 'get_user_settings_store', return_value=mock_settings_store
    ):
        with patch.object(Settings, 'from_config', return_value=config_settings):
            merged_settings = await user_auth.get_user_settings()

    assert merged_settings is not None
    merged_mcp = _sdk_mcp_config(merged_settings)
    assert (
        merged_settings.agent_settings.llm.model
        == 'anthropic/claude-sonnet-4-5-20250929'
    )
    assert merged_mcp is not None
    assert len(merged_mcp.mcpServers) == 2
    assert 'config' in merged_mcp.mcpServers
    assert 'frontend' in merged_mcp.mcpServers


@pytest.mark.asyncio
async def test_user_auth_caching_behavior():
    """Test that user auth caches the merged settings correctly."""
    config_settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='config-model'),
            mcp_config=MCPConfig(
                mcpServers={
                    'config': RemoteMCPServer(
                        url='http://config-server.com', transport='sse'
                    )
                }
            ),
        ),
    )

    stored_settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='anthropic/claude-sonnet-4-5-20250929'),
            mcp_config=MCPConfig(
                mcpServers={
                    'frontend': RemoteMCPServer(
                        url='http://frontend-server.com', transport='sse'
                    )
                }
            ),
        ),
    )

    user_auth = DefaultUserAuth()

    mock_settings_store = AsyncMock(spec=FileSettingsStore)
    mock_settings_store.load.return_value = stored_settings

    with patch.object(
        user_auth, 'get_user_settings_store', return_value=mock_settings_store
    ):
        with patch.object(
            Settings, 'from_config', return_value=config_settings
        ) as mock_from_config:
            settings1 = await user_auth.get_user_settings()
            settings2 = await user_auth.get_user_settings()

    assert settings1 is settings2
    assert len(_sdk_mcp_config(settings1).mcpServers) == 2
    mock_settings_store.load.assert_called_once()
    mock_from_config.assert_called_once()


@pytest.mark.asyncio
async def test_user_auth_no_stored_settings():
    """Test behavior when no settings are stored (first time user)."""
    user_auth = DefaultUserAuth()

    # Mock settings store to return None (no stored settings)
    mock_settings_store = AsyncMock(spec=FileSettingsStore)
    mock_settings_store.load.return_value = None

    with patch.object(
        user_auth, 'get_user_settings_store', return_value=mock_settings_store
    ):
        settings = await user_auth.get_user_settings()

    # Should return None when no settings are stored
    assert settings is None
