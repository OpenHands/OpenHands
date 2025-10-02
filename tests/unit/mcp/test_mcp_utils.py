import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module, not the functions directly to avoid circular imports
import openhands.mcp.utils
from openhands.core.config.mcp_config import MCPSSEServerConfig, MCPStdioServerConfig
from openhands.events.action.mcp import MCPAction
from openhands.events.observation.mcp import MCPObservation


@pytest.mark.asyncio
async def test_create_mcp_clients_empty():
    """Test creating MCP clients with empty server list."""
    clients = await openhands.mcp.utils.create_mcp_clients([], [])
    assert clients == []


@pytest.mark.asyncio
@patch('openhands.mcp.utils.MCPClient')
async def test_create_mcp_clients_success(mock_mcp_client):
    """Test successful creation of MCP clients."""
    # Setup mock
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance
    mock_client_instance.connect_http = AsyncMock()

    # Test with two servers
    server_configs = [
        MCPSSEServerConfig(url='http://server1:8080'),
        MCPSSEServerConfig(url='http://server2:8080', api_key='test-key'),
    ]

    clients = await openhands.mcp.utils.create_mcp_clients(server_configs, [])

    # Verify
    assert len(clients) == 2
    assert mock_mcp_client.call_count == 2

    # Check that connect_http was called with correct parameters
    mock_client_instance.connect_http.assert_any_call(
        server_configs[0], conversation_id=None
    )
    mock_client_instance.connect_http.assert_any_call(
        server_configs[1], conversation_id=None
    )


@pytest.mark.asyncio
@patch('openhands.mcp.utils.MCPClient')
async def test_create_mcp_clients_connection_failure(mock_mcp_client):
    """Test handling of connection failures when creating MCP clients."""
    # Setup mock
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance

    # First connection succeeds, second fails
    mock_client_instance.connect_http.side_effect = [
        None,  # Success
        Exception('Connection failed'),  # Failure
    ]

    server_configs = [
        MCPSSEServerConfig(url='http://server1:8080'),
        MCPSSEServerConfig(url='http://server2:8080'),
    ]

    clients = await openhands.mcp.utils.create_mcp_clients(server_configs, [])

    # Verify only one client was successfully created
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
@patch('openhands.mcp.utils.shutil.which')
async def test_create_mcp_clients_stdio_success(mock_which, mock_mcp_client):
    """Test successful creation of MCP clients with stdio servers."""
    # Setup mock - make both commands appear available
    mock_which.return_value = '/usr/bin/command'  # Mock that commands exist

    # Setup mock
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance
    mock_client_instance.connect_stdio = AsyncMock()

    # Test with stdio servers
    stdio_server_configs = [
        MCPStdioServerConfig(
            name='test-server-1',
            command='python',
            args=['-m', 'server1'],
            env={'DEBUG': 'true'},
        ),
        MCPStdioServerConfig(
            name='test-server-2',
            command='node',
            args=['server2.js'],
            env={'NODE_ENV': 'development'},
        ),
    ]

    clients = await openhands.mcp.utils.create_mcp_clients(
        [], [], stdio_servers=stdio_server_configs
    )

    # Verify
    assert len(clients) == 2
    assert mock_mcp_client.call_count == 2

    # Check that connect_stdio was called with correct parameters
    mock_client_instance.connect_stdio.assert_any_call(stdio_server_configs[0])
    mock_client_instance.connect_stdio.assert_any_call(stdio_server_configs[1])


@pytest.mark.asyncio
@patch('openhands.mcp.utils.MCPClient')
async def test_create_mcp_clients_stdio_connection_failure(mock_mcp_client):
    """Test handling of stdio connection failures when creating MCP clients."""
    # Setup mock
    mock_client_instance = AsyncMock()
    mock_mcp_client.return_value = mock_client_instance

    # First connection succeeds, second fails
    mock_client_instance.connect_stdio.side_effect = [
        None,  # Success
        Exception('Stdio connection failed'),  # Failure
    ]

    stdio_server_configs = [
        MCPStdioServerConfig(name='server1', command='python'),
        MCPStdioServerConfig(name='server2', command='invalid_command'),
    ]

    clients = await openhands.mcp.utils.create_mcp_clients(
        [], [], stdio_servers=stdio_server_configs
    )

    # Verify only one client was successfully created
    assert len(clients) == 1


@pytest.mark.asyncio
@patch('openhands.mcp.utils.create_mcp_clients')
async def test_fetch_mcp_tools_from_config_with_stdio(mock_create_clients):
    """Test fetching MCP tools with stdio servers enabled."""
    from openhands.core.config.mcp_config import MCPConfig

    # Setup mock clients
    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.to_param.return_value = {'function': {'name': 'stdio_tool'}}
    mock_client.tools = [mock_tool]
    mock_create_clients.return_value = [mock_client]

    # Create config with stdio servers
    mcp_config = MCPConfig(
        stdio_servers=[MCPStdioServerConfig(name='test-server', command='python')]
    )

    # Test with use_stdio=True
    tools = await openhands.mcp.utils.fetch_mcp_tools_from_config(
        mcp_config, conversation_id='test-conv', use_stdio=True
    )

    # Verify
    assert len(tools) == 1
    assert tools[0] == {'function': {'name': 'stdio_tool'}}

    # Verify create_mcp_clients was called with stdio servers
    mock_create_clients.assert_called_once_with(
        [], [], 'test-conv', mcp_config.stdio_servers
    )


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
