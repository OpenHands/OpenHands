import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module, not the functions directly to avoid circular imports
import openhands.mcp.utils
from openhands.core.config.mcp_config import (
    MCPConfig,
    RemoteMCPServer,
    StdioMCPServer,
)
from openhands.events.action.mcp import MCPAction
from openhands.events.observation.mcp import MCPObservation


@pytest.mark.asyncio
async def test_create_mcp_clients_empty():
    """Test creating MCP clients with empty config."""
    clients = await openhands.mcp.utils.create_mcp_clients(MCPConfig(mcpServers={}))
    assert clients == []


@pytest.mark.asyncio
@patch('openhands.mcp.utils.MCPClient')
async def test_create_mcp_clients_success(mock_mcp_client):
    """Test successful creation of MCP clients."""
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance
    mock_client_instance.connect_http = AsyncMock()

    s1 = RemoteMCPServer(url='http://server1:8080', transport='sse')
    s2 = RemoteMCPServer(url='http://server2:8080', transport='sse', auth='test-key')
    mcp_config = MCPConfig(mcpServers={'s1': s1, 's2': s2})

    clients = await openhands.mcp.utils.create_mcp_clients(mcp_config)

    assert len(clients) == 2
    assert mock_mcp_client.call_count == 2


@pytest.mark.asyncio
@patch('openhands.mcp.utils.MCPClient')
async def test_create_mcp_clients_connection_failure(mock_mcp_client):
    """Test handling of connection failures when creating MCP clients."""
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance

    mock_client_instance.connect_http.side_effect = [
        None,
        Exception('Connection failed'),
    ]

    s1 = RemoteMCPServer(url='http://server1:8080', transport='sse')
    s2 = RemoteMCPServer(url='http://server2:8080', transport='sse')
    mcp_config = MCPConfig(mcpServers={'s1': s1, 's2': s2})

    clients = await openhands.mcp.utils.create_mcp_clients(mcp_config)

    assert len(clients) == 1


def test_convert_mcp_clients_to_tools_empty():
    """Test converting empty MCP clients list to tools."""
    tools = openhands.mcp.utils.convert_mcp_clients_to_tools(None)
    assert tools == []

    tools = openhands.mcp.utils.convert_mcp_clients_to_tools([])
    assert tools == []


def test_convert_mcp_clients_to_tools():
    """Test converting MCP clients to tools."""
    # Create mock clients with tools
    mock_client1 = MagicMock()
    mock_client2 = MagicMock()

    # Create mock tools
    mock_tool1 = MagicMock()
    mock_tool1.to_param.return_value = {'function': {'name': 'tool1'}}

    mock_tool2 = MagicMock()
    mock_tool2.to_param.return_value = {'function': {'name': 'tool2'}}

    mock_tool3 = MagicMock()
    mock_tool3.to_param.return_value = {'function': {'name': 'tool3'}}

    # Set up the clients with their tools
    mock_client1.tools = [mock_tool1, mock_tool2]
    mock_client2.tools = [mock_tool3]

    # Convert to tools
    tools = openhands.mcp.utils.convert_mcp_clients_to_tools(
        [mock_client1, mock_client2]
    )

    # Verify
    assert len(tools) == 3
    assert tools[0] == {'function': {'name': 'tool1'}}
    assert tools[1] == {'function': {'name': 'tool2'}}
    assert tools[2] == {'function': {'name': 'tool3'}}


@pytest.mark.asyncio
async def test_call_tool_mcp_no_clients():
    """Test calling MCP tool with no clients."""
    action = MCPAction(name='test_tool', arguments={'arg1': 'value1'})

    with pytest.raises(ValueError, match='No MCP clients found'):
        await openhands.mcp.utils.call_tool_mcp([], action)


@pytest.mark.asyncio
async def test_call_tool_mcp_no_matching_client():
    """Test calling MCP tool with no matching client."""
    # Create mock client without the requested tool
    mock_client = MagicMock()
    mock_client.tools = [MagicMock(name='other_tool')]

    action = MCPAction(name='test_tool', arguments={'arg1': 'value1'})

    with pytest.raises(ValueError, match='No matching MCP agent found for tool name'):
        await openhands.mcp.utils.call_tool_mcp([mock_client], action)


@pytest.mark.asyncio
async def test_call_tool_mcp_success():
    """Test successful MCP tool call."""
    # Create mock client with the requested tool
    mock_client = MagicMock()
    mock_tool = MagicMock()
    # Set the name attribute properly for the tool
    mock_tool.name = 'test_tool'
    mock_client.tools = [mock_tool]

    # Setup response
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {'result': 'success'}

    # Setup call_tool method
    mock_client.call_tool = AsyncMock(return_value=mock_response)

    action = MCPAction(name='test_tool', arguments={'arg1': 'value1'})

    # Call the function
    observation = await openhands.mcp.utils.call_tool_mcp([mock_client], action)

    # Verify
    assert isinstance(observation, MCPObservation)
    assert json.loads(observation.content) == {'result': 'success'}
    mock_client.call_tool.assert_called_once_with('test_tool', {'arg1': 'value1'})


@pytest.mark.asyncio
@patch('openhands.mcp.utils.MCPClient')
async def test_create_mcp_clients_stdio_success(mock_mcp_client):
    """Test successful creation of MCP clients with stdio servers."""
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance
    mock_client_instance.connect_stdio = AsyncMock()

    mcp_config = MCPConfig(
        mcpServers={
            'test-server-1': StdioMCPServer(
                command='python',
                args=['-m', 'server1'],
                env={'DEBUG': 'true'},
            ),
            'test-server-2': StdioMCPServer(
                command='node',
                args=['server2.js'],
                env={'NODE_ENV': 'development'},
            ),
        }
    )

    clients = await openhands.mcp.utils.create_mcp_clients(mcp_config)

    assert len(clients) == 2
    assert mock_mcp_client.call_count == 2


@pytest.mark.asyncio
@patch('openhands.mcp.utils.MCPClient')
async def test_create_mcp_clients_stdio_connection_failure(mock_mcp_client):
    """Test handling of stdio connection failures when creating MCP clients."""
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance

    mock_client_instance.connect_stdio.side_effect = [
        None,
        Exception('Stdio connection failed'),
    ]

    mcp_config = MCPConfig(
        mcpServers={
            'server1': StdioMCPServer(command='python'),
            'server2': StdioMCPServer(command='invalid_command'),
        }
    )

    clients = await openhands.mcp.utils.create_mcp_clients(mcp_config)

    assert len(clients) == 1


@pytest.mark.asyncio
@patch('openhands.mcp.utils.create_mcp_clients')
async def test_fetch_mcp_tools_from_config_with_stdio(mock_create_clients):
    """Test fetching MCP tools with stdio servers."""
    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.to_param.return_value = {'function': {'name': 'stdio_tool'}}
    mock_client.tools = [mock_tool]
    mock_create_clients.return_value = [mock_client]

    mcp_config = MCPConfig(mcpServers={'test-server': StdioMCPServer(command='python')})

    tools = await openhands.mcp.utils.fetch_mcp_tools_from_config(
        mcp_config, conversation_id='test-conv'
    )

    assert len(tools) == 1
    assert tools[0] == {'function': {'name': 'stdio_tool'}}

    mock_create_clients.assert_called_once_with(mcp_config, 'test-conv')


@pytest.mark.asyncio
async def test_call_tool_mcp_stdio_client():
    """Test calling MCP tool on a stdio client."""
    # Create mock stdio client with the requested tool
    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = 'stdio_test_tool'
    mock_client.tools = [mock_tool]

    # Setup response
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        'result': 'stdio_success',
        'data': 'test_data',
    }

    # Setup call_tool method
    mock_client.call_tool = AsyncMock(return_value=mock_response)

    action = MCPAction(name='stdio_test_tool', arguments={'input': 'test_input'})

    # Call the function
    observation = await openhands.mcp.utils.call_tool_mcp([mock_client], action)

    # Verify
    assert isinstance(observation, MCPObservation)
    assert json.loads(observation.content) == {
        'result': 'stdio_success',
        'data': 'test_data',
    }
    mock_client.call_tool.assert_called_once_with(
        'stdio_test_tool', {'input': 'test_input'}
    )
