from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.core.config.openhands_config import OpenHandsConfig

from openhands.core.config.mcp_config import (
    MCPSHTTPServerConfig,
    MCPStdioServerConfig,
    OpenHandsMCPConfig,
)
from openhands.core.logger import openhands_logger as logger


# We opt for Streamable HTTP over SSE connection to the main app server's MCP
# Reasoning:
# 1. Better performance over SSE
# 2. Allows stateless MCP client connections, essential for distributed server environments
#
# The second point is very important - any long lived stateful connections (like SSE) will
# require bespoke implementation to make sure all subsequent requests hit the same replica. It is
# also not resistant to replica pod restarts (it will kill the connection and there's no recovering from it)
# NOTE: these details are specific to the MCP protocol
class SaaSOpenHandsMCPConfig(OpenHandsMCPConfig):
    @staticmethod
    def create_default_mcp_server_config(
        host: str, config: 'OpenHandsConfig', user_id: str | None = None
    ) -> tuple[MCPSHTTPServerConfig | None, list[MCPStdioServerConfig]]:
        """
        Create a default MCP server configuration.

        Args:
            host: Host string
            config: OpenHandsConfig
        Returns:
            A tuple containing the default SSE server configuration and a list of MCP stdio server configurations
        """
        from storage.api_key_store import ApiKeyStore

        api_key_store = ApiKeyStore.get_instance()
        if user_id:
            api_key = api_key_store.retrieve_mcp_api_key(user_id)

            if not api_key:
                api_key = api_key_store.create_api_key(user_id, 'MCP_API_KEY', None)

            if not api_key:
                logger.error(f'Could not provision MCP API Key for user: {user_id}')
                return None, []

            return MCPSHTTPServerConfig(
                url=f'https://{host}/mcp/mcp', api_key=api_key
            ), []
        return None, []
