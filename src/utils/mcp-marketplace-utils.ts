import { MCPServerConfig } from "#/types/mcp-server";
import type {
  MCPAuthenticationConfig,
  MCPOAuthClientAuthMethod,
} from "#/types/mcp-auth";
import type {
  IntegrationAuthConfig,
  IntegrationCatalogEntry as MarketplaceEntry,
  IntegrationConnectionOption,
  IntegrationOAuthConfig,
  IntegrationTransport,
} from "@openhands/extensions/integrations";

export type { MarketplaceEntry };

type McpIntegrationAuthConfig = IntegrationAuthConfig & {
  credentialSecretName?: string;
  saveCredentialAsSecretByDefault?: boolean;
};

export type McpMarketplaceConnectionOption = Omit<
  IntegrationConnectionOption,
  "auth" | "provider" | "transport"
> & {
  provider: "mcp";
  transport: IntegrationTransport;
  auth: McpIntegrationAuthConfig;
};

export function getMcpConnectionOptions(
  entry: MarketplaceEntry,
): McpMarketplaceConnectionOption[] {
  return entry.connectionOptions.filter(
    (option): option is McpMarketplaceConnectionOption =>
      option.provider === "mcp" && !!option.transport,
  );
}

export function getDefaultMcpConnectionOption(
  entry: MarketplaceEntry,
): McpMarketplaceConnectionOption | undefined {
  return getMcpConnectionOptions(entry)[0];
}

function isLocallyInstallableMcpOption(
  option: McpMarketplaceConnectionOption,
): boolean {
  if (option.auth.strategy !== "oauth2") return true;

  const oauth = option.auth.oauth;
  if (!oauth) return false;

  // Local agent-server installs only support OAuth flows initiated by the MCP
  // server itself through fastmcp. Catalog entries that specify provider OAuth
  // endpoints still need the hosted integration-auth flow and should not be
  // exposed as locally installable static MCP configs.
  return (
    !oauth.authorizationUrl &&
    !oauth.tokenUrl &&
    !oauth.registrationUrl &&
    !oauth.additionalAuthorizationParams &&
    !oauth.additionalTokenParams
  );
}

export function getInstallableMcpConnectionOption(
  entry: MarketplaceEntry,
): McpMarketplaceConnectionOption | undefined {
  return getMcpConnectionOptions(entry).find(isLocallyInstallableMcpOption);
}

export function getDefaultMcpTransport(
  entry: MarketplaceEntry,
): IntegrationTransport | undefined {
  return getDefaultMcpConnectionOption(entry)?.transport;
}

function toMcpOAuthClientAuthMethod(
  value: IntegrationOAuthConfig["clientAuthentication"] | undefined,
): MCPOAuthClientAuthMethod | undefined {
  switch (value) {
    case "none":
      return "none";
    case "body":
      return "client_secret_post";
    case "basic":
      return "client_secret_basic";
    default:
      return undefined;
  }
}

export function getMcpOAuthAuthenticationConfig(
  option: McpMarketplaceConnectionOption,
): MCPAuthenticationConfig | undefined {
  if (option.auth.strategy !== "oauth2") return undefined;
  const authentication: MCPAuthenticationConfig = { type: "oauth" };
  const clientAuthMethod = toMcpOAuthClientAuthMethod(
    option.auth.oauth?.clientAuthentication,
  );
  if (clientAuthMethod) {
    authentication.client_auth_method = clientAuthMethod;
  }
  if (option.auth.oauth?.scopes?.length) {
    authentication.scopes = option.auth.oauth.scopes;
  }
  return Object.keys(authentication).length > 1 ? authentication : undefined;
}

export function getMcpMarketplaceCatalog(
  catalog: MarketplaceEntry[],
): MarketplaceEntry[] {
  return catalog.filter((entry) => !!getDefaultMcpConnectionOption(entry));
}

const tryUrl = (raw: string): URL | null => {
  try {
    return new URL(raw);
  } catch {
    return null;
  }
};

/**
 * Loose URL match that ignores query strings, trailing slashes, and
 * default ports. We want clicking "Linear" to flag the entry as
 * installed even if the user pasted the URL with extra trailing slash
 * or a different port-equivalent variant.
 *
 * Defensive against runtime data that doesn't match the static type:
 * if either input is not a string (e.g. parsed from an older settings
 * blob), we fall through the URL parsing path and the safe trim
 * fallback below, never calling `.replace` on undefined.
 */
export function urlsMatch(a: unknown, b: unknown): boolean {
  const aStr = typeof a === "string" ? a : "";
  const bStr = typeof b === "string" ? b : "";
  if (!aStr || !bStr) return false;
  const left = tryUrl(aStr);
  const right = tryUrl(bStr);
  if (!left || !right) {
    return aStr.replace(/\/+$/, "") === bStr.replace(/\/+$/, "");
  }
  return (
    left.protocol === right.protocol &&
    left.host === right.host &&
    left.pathname.replace(/\/+$/, "") === right.pathname.replace(/\/+$/, "")
  );
}

/**
 * Decide whether a marketplace template is already represented by one
 * of the installed MCP servers. Used to render an "Installed" badge on
 * the marketplace tile. Returns the first matching server, or null.
 */
export function findInstalledMatch(
  transport: IntegrationTransport,
  servers: MCPServerConfig[],
): MCPServerConfig | null {
  return (
    servers.find((server) => transportMatchesServer(transport, server)) ?? null
  );
}

export function findInstalledEntryMatch(
  entry: MarketplaceEntry,
  servers: MCPServerConfig[],
): MCPServerConfig | null {
  for (const option of getMcpConnectionOptions(entry)) {
    const match = findInstalledMatch(option.transport, servers);
    if (match) return match;
  }
  return null;
}

function transportMatchesServer(
  transport: IntegrationTransport,
  server: MCPServerConfig,
): boolean {
  if (transport.kind === "shttp") {
    const tplUrl = transport.url;
    return (
      server.type === "shttp" && !!server.url && urlsMatch(server.url, tplUrl)
    );
  }

  if (transport.kind === "sse") {
    const tplUrl = transport.url;
    return (
      server.type === "sse" && !!server.url && urlsMatch(server.url, tplUrl)
    );
  }

  // stdio: match on the registered server name.
  return server.type === "stdio" && server.name === transport.serverName;
}

function normalize(query: string): string {
  return query.trim().toLowerCase();
}

/**
 * Case-insensitive substring match against the catalog entry's
 * user-visible identity (name, description, id, keywords). Empty
 * queries always match.
 */
export function getMarketplaceEntriesByPopularity(
  catalog: MarketplaceEntry[],
): MarketplaceEntry[] {
  return catalog
    .map((entry, index) => ({ entry, index }))
    .sort((a, b) => {
      const byPopularity =
        (b.entry.popularityRank ?? 0) - (a.entry.popularityRank ?? 0);
      return byPopularity || a.index - b.index;
    })
    .map(({ entry }) => entry);
}

export function getMarketplaceEntryById(
  id: string,
  catalog: MarketplaceEntry[],
): MarketplaceEntry | undefined {
  return catalog.find((entry) => entry.id === id);
}

export function marketplaceEntryMatchesQuery(
  entry: MarketplaceEntry,
  rawQuery: string,
): boolean {
  const q = normalize(rawQuery);
  if (!q) return true;
  const haystack = [
    entry.name,
    entry.description,
    entry.id,
    ...(entry.keywords ?? []),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}

/**
 * Search match for an installed (already-configured) server. We
 * search the server's own identifying fields and — if it's a catalog
 * entry — its catalog name/keywords too, so typing "Slack" matches
 * the installed Slack tile even though the persisted server is just
 * `{ type: "stdio", name: "slack", ... }`.
 */
export function installedServerMatchesQuery(
  server: MCPServerConfig,
  catalogEntry: MarketplaceEntry | undefined,
  rawQuery: string,
): boolean {
  const q = normalize(rawQuery);
  if (!q) return true;
  const haystack = [
    server.type,
    "name" in server ? server.name : undefined,
    "command" in server ? server.command : undefined,
    "args" in server ? server.args?.join(" ") : undefined,
    "url" in server ? server.url : undefined,
    catalogEntry?.name,
    catalogEntry?.description,
    catalogEntry?.id,
    ...(catalogEntry?.keywords ?? []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}

/**
 * Look up the catalog entry that best matches an installed server.
 * Mirrors the lookup used in `installed-server-card.tsx` for
 * rendering the friendly icon.
 */
export function findCatalogEntryForServer(
  server: MCPServerConfig,
  catalog: MarketplaceEntry[],
): MarketplaceEntry | undefined {
  return catalog.find((entry) => {
    // Check every MCP option rather than only the default. Some unified
    // integration entries default to OAuth-hosted MCP while still exposing
    // an API/stdio option; existing installed servers should match either.
    return getMcpConnectionOptions(entry).some((option) =>
      transportMatchesServer(option.transport, server),
    );
  });
}
