import type { IntegrationCatalogEntry as MarketplaceEntry } from "@openhands/extensions/integrations";
import type { MCPServerConfig } from "#/types/mcp-server";

function configuredServerName(server: MCPServerConfig): string | undefined {
  const name = server.name?.trim();
  return name ? name : undefined;
}

/**
 * Marketplace installs persist the catalog slug as the MCP server name so the
 * backend has a stable reference key. Treat that implicit slug as metadata,
 * while still honoring names the user entered explicitly.
 */
export function getInstalledServerTitle(
  server: MCPServerConfig,
  catalog?: MarketplaceEntry,
): string {
  const name = configuredServerName(server);
  if (name && name !== catalog?.id) {
    return name;
  }

  if (catalog?.name) {
    return catalog.name;
  }

  if (server.type === "stdio") {
    return name ?? server.command ?? "";
  }

  return server.url ?? "";
}
