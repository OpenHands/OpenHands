import asyncio
from unittest import mock

import pytest

from openhands.core.config.mcp_config import MCPConfig, RemoteMCPServer
from openhands.mcp import MCPClient, create_mcp_clients, fetch_mcp_tools_from_config


@pytest.mark.asyncio
async def test_sse_connection_timeout():
    """Test that SSE connection timeout is handled gracefully."""
    mock_client = mock.MagicMock(spec=MCPClient)

    async def mock_connect_http(*args, **kwargs):
        await asyncio.sleep(0.1)
        raise asyncio.TimeoutError('Connection timed out')

    mock_client.connect_http.side_effect = mock_connect_http
    mock_client.disconnect = mock.AsyncMock()

    with mock.patch('openhands.mcp.utils.MCPClient', return_value=mock_client):
        mcp_config = MCPConfig(
            mcpServers={
                's1': RemoteMCPServer(url='http://server1:8080', transport='sse'),
                's2': RemoteMCPServer(url='http://server2:8080', transport='sse'),
            }
        )

        clients = await create_mcp_clients(mcp_config)

        assert len(clients) == 0
        assert mock_client.connect_http.call_count == 2


@pytest.mark.asyncio
async def test_fetch_mcp_tools_with_timeout():
    """Test that fetch_mcp_tools_from_config handles timeouts gracefully."""
    mock_config = MCPConfig(
        mcpServers={
            's1': RemoteMCPServer(url='http://server1:8080', transport='sse'),
        }
    )

    with mock.patch('openhands.mcp.utils.create_mcp_clients', return_value=[]):
        tools = await fetch_mcp_tools_from_config(mock_config, None)
        assert tools == []


@pytest.mark.asyncio
async def test_mixed_connection_results():
    """Test that fetch_mcp_tools_from_config returns tools even when some connections fail."""
    mock_config = MCPConfig(
        mcpServers={
            's1': RemoteMCPServer(url='http://server1:8080', transport='sse'),
            's2': RemoteMCPServer(url='http://server2:8080', transport='sse'),
        }
    )

    # Create a successful client
    successful_client = mock.MagicMock(spec=MCPClient)

    # Create a mock tool with a to_param method that returns a tool dictionary
    mock_tool = mock.MagicMock()
    mock_tool.name = 'mock_tool'
    mock_tool.to_param.return_value = {
        'type': 'function',
        'function': {
            'name': 'mock_tool',
            'description': 'A mock tool for testing',
            'parameters': {},
        },
    }

    # Set the client's tools
    successful_client.tools = [mock_tool]

    # Mock create_mcp_clients to return our successful client
    with mock.patch(
        'openhands.mcp.utils.create_mcp_clients', return_value=[successful_client]
    ):
        # Call fetch_mcp_tools_from_config
        tools = await fetch_mcp_tools_from_config(mock_config, None)

        # Verify that tools were returned
        assert len(tools) > 0
        assert tools[0]['function']['name'] == 'mock_tool'
