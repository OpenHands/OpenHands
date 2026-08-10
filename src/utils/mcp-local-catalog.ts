import type { IntegrationCatalogEntry } from "@openhands/extensions/integrations";

/**
 * Heimdall / fork-local MCP marketplace entries that are not (yet) in
 * `@openhands/extensions`. Merged by {@link getMcpMarketplaceCatalog}.
 *
 * Appwrite local stdio server:
 * https://appwrite.io/docs/tooling/ai/mcp-servers
 * https://appwrite.io/docs/advanced/self-hosting/mcp
 *
 * Plane local stdio server:
 * https://github.com/makeplane/plane-mcp-server
 * https://developers.plane.so/api-reference/introduction
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
              helperText:
                "Project ID from the Appwrite Console project settings.",
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
              helperLink: "https://appwrite.io/docs/advanced/self-hosting/mcp",
            },
          ],
        },
        auth: {
          strategy: "api_key",
        },
      },
    ],
  },
  {
    id: "plane",
    name: "Plane",
    description:
      "Manage Plane work items, cycles, modules, and projects via the official MCP server.",
    docsUrl: "https://developers.plane.so/api-reference/introduction",
    appUrl: "https://plane.so",
    iconBg: "#0D0E11",
    logoUrl: "https://cdn.simpleicons.org/plane/3F76FF",
    keywords: [
      "plane",
      "project management",
      "issues",
      "work items",
      "cycles",
      "modules",
      "agile",
    ],
    popularityRank: 84,
    installHint:
      "Provide your Plane API key, workspace slug, and base URL (self-hosted or https://api.plane.so). Runs via uvx plane-mcp-server stdio.",
    connectionOptions: [
      {
        id: "api",
        provider: "mcp",
        transport: {
          kind: "stdio",
          serverName: "plane",
          command: "uvx",
          args: ["plane-mcp-server", "stdio"],
          envFields: [
            {
              key: "PLANE_API_KEY",
              label: "API key",
              type: "password",
              placeholder: "plane_api_…",
              required: true,
              helperText:
                "Personal access token from Plane Profile Settings → Personal Access Tokens.",
              helperLink:
                "https://developers.plane.so/api-reference/introduction",
            },
            {
              key: "PLANE_WORKSPACE_SLUG",
              label: "Workspace slug",
              type: "text",
              placeholder: "my-workspace",
              required: true,
              helperText:
                "Workspace slug from your Plane URL (e.g. heimdall in plane.example.com/heimdall).",
              helperLink:
                "https://developers.plane.so/api-reference/introduction",
            },
            {
              key: "PLANE_BASE_URL",
              label: "Base URL",
              type: "text",
              placeholder: "https://plane.example.com",
              required: false,
              helperText:
                "Self-hosted Plane URL, or leave blank for Plane Cloud (https://api.plane.so).",
              helperLink: "https://github.com/makeplane/plane-mcp-server",
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
