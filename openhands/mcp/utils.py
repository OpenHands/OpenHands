import asyncio
import json
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.controller.agent import Agent
    from openhands.memory.memory import Memory


from mcp import McpError

from openhands.core.config.mcp_config import (
    MCPConfig,
    MCPSHTTPServerConfig,
    MCPSSEServerConfig,
    MCPStdioServerConfig,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.mcp import MCPAction
from openhands.events.observation.mcp import MCPObservation
from openhands.events.observation.observation import Observation
from openhands.mcp.client import MCPClient
from openhands.mcp.error_collector import mcp_error_collector
from openhands.runtime.base import Runtime
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime
from openhands.utils._redact_compat import (
    redact_text_secrets,
    redact_url_params,
    sanitize_config,
)


def convert_mcp_clients_to_tools(mcp_clients: list[MCPClient] | None) -> list[dict]:
    """Converts a list of MCPClient instances to ChatCompletionToolParam format
    that can be used by CodeActAgent.

    Args:
        mcp_clients: List of MCPClient instances or None

    Returns:
        List of dicts of tools ready to be used by CodeActAgent
    """
    if mcp_clients is None:
        logger.warning('mcp_clients is None, returning empty list')
        return []

    all_mcp_tools = []
    try:
        for client in mcp_clients:
            # Each MCPClient has an mcp_clients property that is a ToolCollection
            # The ToolCollection has a to_params method that converts tools to ChatCompletionToolParam format
            for tool in client.tools:
                mcp_tools = tool.to_param()
                all_mcp_tools.append(mcp_tools)
    except Exception as e:
        error_msg = f'Error in convert_mcp_clients_to_tools: {e}'
        logger.error(error_msg)
        mcp_error_collector.add_error(
            server_name='general',
            server_type='conversion',
            error_message=error_msg,
            exception_details=str(e),
        )
        return []
    return all_mcp_tools


async def create_mcp_clients(
    sse_servers: list[MCPSSEServerConfig],
    shttp_servers: list[MCPSHTTPServerConfig],
    conversation_id: str | None = None,
    stdio_servers: list[MCPStdioServerConfig] | None = None,
) -> list[MCPClient]:
    import sys

    # Skip MCP clients on Windows
    if sys.platform == 'win32':
        logger.info(
            'MCP functionality is disabled on Windows, skipping client creation'
        )
        return []

    if stdio_servers is None:
        stdio_servers = []

    servers: list[MCPSSEServerConfig | MCPSHTTPServerConfig | MCPStdioServerConfig] = [
        *sse_servers,
        *shttp_servers,
        *stdio_servers,
    ]

    if not servers:
        return []

    mcp_clients = []

    for server in servers:
        if isinstance(server, MCPStdioServerConfig):
            # Validate that the command exists before connecting
            if not shutil.which(server.command):
                logger.error(
                    f'Skipping MCP stdio server "{server.name}": command "{server.command}" not found. '
                    f'Please install {server.command} or remove this server from your configuration.'
                )
                continue

            logger.info(
                f'Initializing MCP agent for {redact_text_secrets(str(server))} with stdio connection...'
            )
            client = MCPClient()
            try:
                await client.connect_stdio(server)

                # Log which tools this specific server provides
                tool_names = [tool.name for tool in client.tools]
                server_name = getattr(
                    server, 'name', f'{server.command} {" ".join(server.args or [])}'
                )
                logger.debug(
                    f'Successfully connected to MCP stdio server {server_name} - '
                    f'provides {len(tool_names)} tools: {tool_names}'
                )

                mcp_clients.append(client)
            except Exception as e:
                # Error is already logged and collected in client.connect_stdio()
                logger.error(
                    f'Failed to connect to {redact_text_secrets(str(server))}: {str(e)}',
                    exc_info=True,
                )
            continue

        is_shttp = isinstance(server, MCPSHTTPServerConfig)

        connection_type = 'SHTTP' if is_shttp else 'SSE'
        logger.info(
            f'Initializing MCP agent for {redact_text_secrets(str(server))} with {connection_type} connection...'
        )
        client = MCPClient()

        # Set server timeout for SHTTP servers
        if isinstance(server, MCPSHTTPServerConfig) and server.timeout is not None:
            client.server_timeout = float(server.timeout)
            logger.debug(f'Set SHTTP server timeout to {server.timeout}s')

        try:
            await client.connect_http(server, conversation_id=conversation_id)

            # Log which tools this specific server provides
            tool_names = [tool.name for tool in client.tools]
            logger.debug(
                f'Successfully connected to MCP STTP server {redact_url_params(server.url)} - '
                f'provides {len(tool_names)} tools: {tool_names}'
            )

            # Only add the client to the list after a successful connection
            mcp_clients.append(client)

        except Exception as e:
            # Error is already logged and collected in client.connect_http()
            logger.error(
                f'Failed to connect to {redact_text_secrets(str(server))}: {str(e)}',
                exc_info=True,
            )

    return mcp_clients


async def fetch_mcp_tools_from_config(
    mcp_config: MCPConfig, conversation_id: str | None = None, use_stdio: bool = False
) -> list[dict]:
    """Retrieves the list of MCP tools from the MCP clients.

    Args:
        mcp_config: The MCP configuration
        conversation_id: Optional conversation ID to associate with the MCP clients
        use_stdio: Whether to use stdio servers for MCP clients, set to True when running from a CLI runtime

    Returns:
        A list of tool dictionaries. Returns an empty list if no connections could be established.
    """
    import sys

    # Skip MCP tools on Windows
    if sys.platform == 'win32':
        logger.info('MCP functionality is disabled on Windows, skipping tool fetching')
        return []

    mcp_clients = []
    mcp_tools = []
    try:
        logger.debug(
            f'Creating MCP clients with config: {sanitize_config(mcp_config.model_dump())}'
        )

        # Create clients - this will fetch tools but not maintain active connections
        mcp_clients = await create_mcp_clients(
            mcp_config.sse_servers,
            mcp_config.shttp_servers,
            conversation_id,
            mcp_config.stdio_servers if use_stdio else [],
        )

        if not mcp_clients:
            logger.debug('No MCP clients were successfully connected')
            return []

        # Convert tools to the format expected by the agent
        mcp_tools = convert_mcp_clients_to_tools(mcp_clients)

    except Exception as e:
        error_msg = f'Error fetching MCP tools: {str(e)}'
        logger.error(error_msg)
        mcp_error_collector.add_error(
            server_name='general',
            server_type='fetch',
            error_message=error_msg,
            exception_details=str(e),
        )
        return []

    logger.debug(f'MCP tools: {mcp_tools}')
    return mcp_tools


async def call_tool_mcp(mcp_clients: list[MCPClient], action: MCPAction) -> Observation:
    """Call a tool on an MCP server and return the observation.

    Args:
        mcp_clients: The list of MCP clients to execute the action on
        action: The MCP action to execute

    Returns:
        The observation from the MCP server
    """
    import sys

    from openhands.events.observation import ErrorObservation

    # Skip MCP tools on Windows
    if sys.platform == 'win32':
        logger.info('MCP functionality is disabled on Windows')
        return ErrorObservation('MCP functionality is not available on Windows')

    if not mcp_clients:
        raise ValueError('No MCP clients found')

    logger.debug(f'MCP action received: {action}')

    # Find the MCP client that has the matching tool name
    matching_client = None
    logger.debug(f'MCP clients: {mcp_clients}')
    logger.debug(f'MCP action name: {action.name}')

    for client in mcp_clients:
        logger.debug(f'MCP client tools: {client.tools}')
        if action.name in [tool.name for tool in client.tools]:
            matching_client = client
            break

    if matching_client is None:
        raise ValueError(f'No matching MCP agent found for tool name: {action.name}')

    logger.debug(f'Matching client: {matching_client}')

    try:
        # Call the tool - this will create a new connection internally
        response = await matching_client.call_tool(action.name, action.arguments)
        logger.debug(f'MCP response: {response}')

        return MCPObservation(
            content=json.dumps(response.model_dump(mode='json')),
            name=action.name,
            arguments=action.arguments,
        )
    except asyncio.TimeoutError:
        # Handle timeout errors specifically
        timeout_val = getattr(matching_client, 'server_timeout', 'unknown')
        logger.error(f'MCP tool {action.name} timed out after {timeout_val}s')
        error_content = json.dumps(
            {
                'isError': True,
                'error': f'Tool "{action.name}" timed out after {timeout_val} seconds',
                'content': [],
            }
        )
        return MCPObservation(
            content=error_content,
            name=action.name,
            arguments=action.arguments,
        )
    except McpError as e:
        # Handle MCP errors by returning an error observation instead of raising
        logger.error(f'MCP error when calling tool {action.name}: {e}')
        error_content = json.dumps({'isError': True, 'error': str(e), 'content': []})
        return MCPObservation(
            content=error_content,
            name=action.name,
            arguments=action.arguments,
        )


async def add_mcp_tools_to_agent(
    agent: 'Agent', runtime: Runtime, memory: 'Memory'
) -> MCPConfig:
    """Add MCP tools to an agent."""
    import sys

    # Skip MCP tools on Windows
    if sys.platform == 'win32':
        logger.info('MCP functionality is disabled on Windows, skipping MCP tools')
        agent.set_mcp_tools([])
        return

    assert runtime.runtime_initialized, (
        'Runtime must be initialized before adding MCP tools'
    )

    extra_stdio_servers = []

    # Add microagent MCP tools if available
    microagent_mcp_configs = memory.get_microagent_mcp_tools()
    for mcp_config in microagent_mcp_configs:
        if mcp_config.sse_servers:
            logger.warning(
                'Microagent MCP config contains SSE servers, it is not yet supported.'
            )

        if mcp_config.stdio_servers:
            for stdio_server in mcp_config.stdio_servers:
                # Check if this stdio server is already in the config
                if stdio_server not in extra_stdio_servers:
                    extra_stdio_servers.append(stdio_server)
                    logger.warning(
                        f'Added microagent stdio server: {stdio_server.name}'
                    )

    # Add the runtime as another MCP server
    updated_mcp_config = runtime.get_mcp_config(extra_stdio_servers)

    # Fetch the MCP tools
    # Only use stdio if run from a CLI runtime
    mcp_tools = await fetch_mcp_tools_from_config(
        updated_mcp_config, use_stdio=isinstance(runtime, CLIRuntime)
    )

    tool_names = [tool['function']['name'] for tool in mcp_tools]
    logger.info(f'Loaded {len(mcp_tools)} MCP tools: {tool_names}')

    # Set the MCP tools on the agent
    agent.set_mcp_tools(mcp_tools)

    return updated_mcp_config
