import asyncio
from typing import Optional

from fastmcp import Client
from fastmcp.client.transports import (
    SSETransport,
    StdioTransport,
    StreamableHttpTransport,
)
from mcp import McpError
from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict, Field

from openhands.core.config.mcp_config import RemoteMCPServer, StdioMCPServer
from openhands.core.logger import openhands_logger as logger
from openhands.mcp.error_collector import mcp_error_collector
from openhands.mcp.tool import MCPClientTool


class MCPClient(BaseModel):
    """A collection of tools that connects to an MCP server and manages available tools through the Model Context Protocol."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Optional[Client] = None
    description: str = 'MCP client tools for server interaction'
    tools: list[MCPClientTool] = Field(default_factory=list)
    tool_map: dict[str, MCPClientTool] = Field(default_factory=dict)
    server_timeout: Optional[float] = None  # Timeout from server config for tool calls

    async def _initialize_and_list_tools(self) -> None:
        """Initialize session and populate tool map."""
        if not self.client:
            raise RuntimeError('Session not initialized.')

        async with self.client:
            tools = await self.client.list_tools()

        # Clear existing tools
        self.tools = []

        # Create proper tool objects for each server tool
        for tool in tools:
            server_tool = MCPClientTool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.inputSchema,
            )
            self.tool_map[tool.name] = server_tool
            self.tools.append(server_tool)

        logger.info(f'Connected to server with tools: {[tool.name for tool in tools]}')

    async def connect_http(
        self,
        server: RemoteMCPServer,
        conversation_id: str | None = None,
        timeout: float = 30.0,
    ):
        """Connect to MCP server using streamable HTTP or SSE transport."""
        server_url = server.url
        api_key = (
            str(server.auth) if server.auth else None
        )  # SDK uses `auth` for bearer token

        if not server_url:
            raise ValueError('Server URL is required.')

        transport_type = server.transport or 'http'

        try:
            headers: dict[str, str] = (
                {k: str(v) for k, v in server.headers.items()} if server.headers else {}
            )
            if api_key:
                headers.update(
                    {
                        'Authorization': f'Bearer {api_key}',
                        's': api_key,
                        'X-Session-API-Key': api_key,
                    }
                )

            if conversation_id:
                headers['X-OpenHands-ServerConversation-ID'] = conversation_id

            transport: SSETransport | StreamableHttpTransport
            if transport_type == 'sse':
                transport = SSETransport(
                    url=server_url,
                    headers=headers or None,
                )
            else:
                transport = StreamableHttpTransport(
                    url=server_url,
                    headers=headers or None,
                )
            self.client = Client(transport, timeout=timeout)

            await self._initialize_and_list_tools()
        except McpError as e:
            error_msg = f'McpError connecting to {server_url}: {e}'
            logger.error(error_msg)
            mcp_error_collector.add_error(
                server_name=server_url,
                server_type=transport_type,
                error_message=error_msg,
                exception_details=str(e),
            )
            raise

        except Exception as e:
            error_msg = f'Error connecting to {server_url}: {e}'
            logger.error(error_msg)
            mcp_error_collector.add_error(
                server_name=server_url,
                server_type=transport_type,
                error_message=error_msg,
                exception_details=str(e),
            )
            raise

    async def connect_stdio(
        self,
        server: StdioMCPServer,
        name: str | None = None,
        timeout: float = 30.0,
    ):
        """Connect to MCP server using stdio transport."""
        server_name = name or f'{server.command} {" ".join(server.args or [])}'
        try:
            transport = StdioTransport(
                command=server.command, args=server.args or [], env=server.env
            )
            self.client = Client(transport, timeout=timeout)
            await self._initialize_and_list_tools()
        except Exception as e:
            error_msg = f'Failed to connect to stdio server {server_name}: {e}'
            logger.error(error_msg)
            mcp_error_collector.add_error(
                server_name=server_name,
                server_type='stdio',
                error_message=error_msg,
                exception_details=str(e),
            )
            raise

    async def call_tool(self, tool_name: str, args: dict) -> CallToolResult:
        """Call a tool on the MCP server with timeout from server configuration.

        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool

        Returns:
            CallToolResult from the MCP server

        Raises:
            asyncio.TimeoutError: If the tool call times out
            ValueError: If the tool is not found
            RuntimeError: If the client session is not available
        """
        if tool_name not in self.tool_map:
            raise ValueError(f'Tool {tool_name} not found.')
        # The MCPClientTool is primarily for metadata; use the session to call the actual tool.
        if not self.client:
            raise RuntimeError('Client session is not available.')

        async with self.client:
            # Use server timeout if configured
            if self.server_timeout is not None:
                return await asyncio.wait_for(
                    self.client.call_tool_mcp(name=tool_name, arguments=args),
                    timeout=self.server_timeout,
                )
            else:
                return await self.client.call_tool_mcp(name=tool_name, arguments=args)
