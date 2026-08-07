import type { IntegrationCatalogEntry } from "@openhands/extensions/integrations";

/**
 * Heimdall / fork-local MCP marketplace entries that are not (yet) in
 * `@openhands/extensions`. Merged by {@link getMcpMarketplaceCatalog}.
 *
 * Appwrite local stdio server:
 * https://appwrite.io/docs/tooling/ai/mcp-servers
 * https://appwrite.io/docs/advanced/self-hosting/mcp
 */
export const LOCAL_MCP_CATALOG: IntegrationCatalogEntry[] = [
  {
    id: "appwrite",
    name: "Appwrite",
    description:
      "Manage Appwrite projects — users, databases, storage, functions, and docs — via the official MCP server.",
    docsUrl: "https://appwrite.io/docs/tooling/ai/mcp-servers",
    appUrl: "https://cloud.appwrite.io",
    iconBg: "#19191C",
    logoUrl: "https://cdn.simpleicons.org/appwrite/FD366E",
    keywords: [
      "appwrite",
      "backend",
      "baas",
      "database",
      "auth",
      "storage",
      "functions",
    ],
    popularityRank: 85,
    installHint:
      "Provide your Appwrite project ID, API key, and endpoint (must end with /v1). Runs via uvx mcp-server-appwrite.",
    connectionOptions: [
      {
        id: "api",
        provider: "mcp",
        transport: {
          kind: "stdio",
          serverName: "appwrite",
          command: "uvx",
          args: ["mcp-server-appwrite"],
          envFields: [
            {
              key: "APPWRITE_API_KEY",
              label: "API key",
              type: "password",
              placeholder: "standard_…",
              required: true,
              helperText:
                "API key from your Appwrite project (Settings → API keys). Enable the scopes you need.",
              helperLink: "https://cloud.appwrite.io/console",
            },
            {
              key: "APPWRITE_PROJECT_ID",
              label: "Project ID",
              type: "text",
              placeholder: "your-project-id",
              required: true,
              helperText: "Project ID from the Appwrite Console project settings.",
              helperLink: "https://cloud.appwrite.io/console",
            },
            {
              key: "APPWRITE_ENDPOINT",
              label: "Endpoint",
              type: "text",
              placeholder: "https://cloud.appwrite.io/v1",
              required: true,
              helperText:
                "API endpoint ending in /v1 (Cloud or self-hosted). Example: https://cloud.appwrite.io/v1",
              helperLink:
                "https://appwrite.io/docs/advanced/self-hosting/mcp",
            },
          ],
        },
        auth: {
          strategy: "api_key",
        },
      },
    ],
  },
];
