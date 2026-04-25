import os

from fastmcp import Client, FastMCP
from fastmcp.client.transports import NpxStdioTransport

from openhands.core.logger import openhands_logger as logger
from openhands.app_server.app_conversation.mcp.mcp import mcp_server

ENABLE_MCP_SEARCH_ENGINE = (
    os.getenv('ENABLE_MCP_SEARCH_ENGINE', 'false').lower() == 'true'
)


def patch_mcp_server():
    if not ENABLE_MCP_SEARCH_ENGINE:
        logger.warning('Tavily search integration is disabled')
        return

    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

    if TAVILY_API_KEY:
        proxy_client = Client(
            transport=NpxStdioTransport(
                package='tavily-mcp@0.2.1', env_vars={'TAVILY_API_KEY': TAVILY_API_KEY}
            )
        )
        proxy_server = FastMCP.as_proxy(proxy_client)

        mcp_server.mount(prefix='tavily', server=proxy_server)
        logger.info('Tavily search integration initialized successfully')
    else:
        logger.warning('Tavily API key not found, skipping search integration')
