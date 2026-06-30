"""Unit tests for MCP test service helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.mcp_config import MCPConfig, RemoteMCPServer, StdioMCPServer
from pydantic import SecretStr

from openhands.app_server.mcp.mcp_test_models import McpServerTransport
from openhands.app_server.mcp.mcp_test_service import (
    McpServerNotFoundError,
    resolve_mcp_server,
)
from openhands.app_server.settings.settings_models import Settings
from openhands.sdk.llm import LLM
from openhands.sdk.settings import OpenHandsAgentSettings


def _settings_with_mcp(mcp_servers: dict) -> Settings:
    return Settings(
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(model='gpt-4', api_key=SecretStr('test-key')),
            mcp_config=MCPConfig(mcpServers=mcp_servers),
        )
    )


def test_resolve_stdio_server():
    settings = _settings_with_mcp(
        {
            'demo': StdioMCPServer(command='python', args=['server.py']),
        }
    )
    resolved = resolve_mcp_server(settings, 'demo')
    assert resolved.transport == McpServerTransport.STDIO
    assert resolved.server_dump['command'] == 'python'


def test_resolve_sse_server():
    settings = _settings_with_mcp(
        {
            'remote': RemoteMCPServer(url='http://localhost:8080/mcp', transport='sse'),
        }
    )
    resolved = resolve_mcp_server(settings, 'remote')
    assert resolved.transport == McpServerTransport.SSE
    assert resolved.url == 'http://localhost:8080/mcp'


def test_resolve_shttp_server():
    settings = _settings_with_mcp(
        {
            'remote': RemoteMCPServer(url='http://localhost:8080/mcp'),
        }
    )
    resolved = resolve_mcp_server(settings, 'remote')
    assert resolved.transport == McpServerTransport.SHTTP


def test_resolve_missing_server_raises():
    settings = _settings_with_mcp({})
    with pytest.raises(McpServerNotFoundError):
        resolve_mcp_server(settings, 'missing')


@pytest.mark.asyncio
async def test_start_test_reuses_running_run():
    from uuid import uuid4

    from fastmcp.mcp_config import MCPConfig, StdioMCPServer
    from pydantic import SecretStr

    from openhands.app_server.mcp.mcp_test_models import (
        McpServerTestRun,
        McpServerTestRunStatus,
    )
    from openhands.app_server.mcp.mcp_test_service import McpTestService
    from openhands.app_server.settings.settings_models import Settings
    from openhands.sdk.llm import LLM
    from openhands.sdk.settings import OpenHandsAgentSettings

    existing = McpServerTestRun(
        id=uuid4(),
        server_id='demo',
        transport=McpServerTransport.STDIO,
        status=McpServerTestRunStatus.RUNNING,
    )
    test_run_service = AsyncMock()
    test_run_service.get_running_test_run.return_value = existing

    user_context = AsyncMock()
    user_context.get_user_id.return_value = 'user-1'

    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(model='gpt-4', api_key=SecretStr('test-key')),
            mcp_config=MCPConfig(
                mcpServers={'demo': StdioMCPServer(command='python', args=['x.py'])}
            ),
        )
    )

    service = McpTestService(
        test_run_service=test_run_service,
        sandbox_service=MagicMock(),
        sandbox_spec_service=MagicMock(),
        httpx_client=MagicMock(),
        user_context=user_context,
    )
    result = await service.start_test('demo', settings)
    assert result.id == existing.id
    test_run_service.create_test_run.assert_not_called()
