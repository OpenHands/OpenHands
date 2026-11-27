// The MCP page now lives at the top-level /mcp route. This file is
// kept for backward-compatibility so the published `MCPSettings`
// library export (see `src/components/settings/index.ts`) and any
// remaining `routes/mcp-settings` import paths keep working.

import MCPPage from "./mcp";

export const MCPSettingsScreen = MCPPage;

export default MCPPage;
