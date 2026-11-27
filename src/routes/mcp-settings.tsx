// The /mcp route redirects to /agents, but the standalone MCP screen
// remains available through the published `MCPSettings` library export and
// existing `routes/mcp-settings` imports.

import { MCPPage } from "./mcp";

export const MCPSettingsScreen = MCPPage;

export default MCPPage;
