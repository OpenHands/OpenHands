import {
  MCPConfig,
  MCPAuthCredential,
  MCPSSEServer,
  MCPSHTTPServer,
  MCPStdioServer,
  SettingsValue,
} from "#/types/settings";

const EMPTY_MCP_CONFIG: MCPConfig = {
  sse_servers: [],
  stdio_servers: [],
  shttp_servers: [],
};

type SdkMcpServerConfig = Record<string, SettingsValue | MCPAuthCredential>;

const MCP_AUTH_STRATEGIES = new Set([
  "none",
  "api_key",
  "bearer",
  "basic",
  "header",
  "oauth2",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function stringRecord(value: unknown): Record<string, string> | undefined {
  if (!isRecord(value)) return undefined;
  const entries = Object.entries(value).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );
  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}

function authCredential(value: unknown): MCPAuthCredential | undefined {
  if (
    !isRecord(value) ||
    typeof value.strategy !== "string" ||
    !MCP_AUTH_STRATEGIES.has(value.strategy)
  ) {
    return undefined;
  }
  return value as MCPAuthCredential;
}

/**
 * Normalize the SDK wire payload to a flat server map.
 *
 * SDK 1.31.x dropped the ``mcpServers`` wrapper from ``model_dump``, so the
 * ``GET /api/v1/settings`` response now returns a flat dict keyed by server
 * name (e.g. ``{shttp: {...}}``) instead of ``{mcpServers: {shttp: {...}}}``.
 * Older persisted settings, and clients that still send the wrapper, are
 * accepted by the SDK's ``_normalize_mcp_config_field``, so we mirror its
 * unwrap-or-pass-through logic here for both directions.
 */
function unwrapMcpServers(
  value: Record<string, unknown>,
): Record<string, unknown> {
  if (
    "mcpServers" in value &&
    value.mcpServers &&
    typeof value.mcpServers === "object"
  ) {
    return value.mcpServers as Record<string, unknown>;
  }
  return value;
}

/**
 * Returns true if ``value`` already matches the frontend's parsed ``MCPConfig``
 * shape (the grouped ``{ sse_servers, stdio_servers, shttp_servers }`` form
 * used internally by the UI and as the placeholder on ``DEFAULT_AGENT_SETTINGS``).
 *
 * The check requires **exactly** the three grouping keys as arrays; a strict
 * 3-key shape avoids misidentifying SDK-format payloads that happen to also
 * carry an empty ``sse_servers``/``stdio_servers``/``shttp_servers`` array
 * (e.g. a defensive frontend caller returning ``EMPTY_MCP_CONFIG`` alongside
 * an SDK-format flat map).
 */
function isParsedMcpConfigShape(value: Record<string, unknown>): boolean {
  const keys = Object.keys(value);
  return (
    keys.length === 3 &&
    Array.isArray(value.sse_servers) &&
    Array.isArray(value.stdio_servers) &&
    Array.isArray(value.shttp_servers)
  );
}

function apiKeyFromAuthorizationHeader(value: unknown): string | undefined {
  if (Array.isArray(value)) {
    return value
      .map(apiKeyFromAuthorizationHeader)
      .find((apiKey) => apiKey !== undefined);
  }

  if (typeof value !== "string" || value.length === 0) return undefined;
  const bearer = value.match(/^Bearer\s+(.+)$/i);
  return bearer ? bearer[1] : value;
}

/**
 * Recover a remote server's API key from either the canonical
 * ``headers.Authorization`` bearer token or the legacy ``auth`` field (kept
 * for back-compat with settings persisted before the header migration).
 */
function apiKeyFromServerConfig(
  serverConfig: Record<string, unknown>,
): string | undefined {
  const { headers } = serverConfig;
  const authorization =
    headers && typeof headers === "object"
      ? ((headers as Record<string, unknown>).Authorization ??
        (headers as Record<string, unknown>).authorization)
      : undefined;
  const headerApiKey = apiKeyFromAuthorizationHeader(authorization);
  if (headerApiKey) return headerApiKey;

  const { auth } = serverConfig;
  if (auth && typeof auth === "object" && !Array.isArray(auth)) {
    const credential = auth as Record<string, unknown>;
    if (
      (credential.strategy === "api_key" || credential.strategy === "bearer") &&
      typeof credential.value === "string"
    ) {
      return credential.value;
    }
    if (credential.strategy === "header") {
      const authHeaders = credential.headers;
      if (authHeaders && typeof authHeaders === "object") {
        return apiKeyFromAuthorizationHeader(
          (authHeaders as Record<string, unknown>).Authorization ??
            (authHeaders as Record<string, unknown>).authorization,
        );
      }
    }
  }
  return typeof auth === "string" && auth !== "oauth" ? auth : undefined;
}

function authorizationHeader(
  headers: Record<string, string> | undefined,
): [string, string] | undefined {
  return Object.entries(headers ?? {}).find(
    ([key]) => key.toLowerCase() === "authorization",
  );
}

function withApiKey(
  auth: MCPAuthCredential | undefined,
  apiKey: string,
): MCPAuthCredential {
  if (auth?.strategy === "api_key" || auth?.strategy === "bearer") {
    return { ...auth, value: apiKey };
  }
  if (auth?.strategy === "header") {
    const current = authorizationHeader(auth.headers ?? undefined);
    if (current) {
      const [key, value] = current;
      const headerValue = /^Bearer\s+/i.test(value)
        ? `Bearer ${apiKey}`
        : apiKey;
      return {
        ...auth,
        headers: { ...auth.headers, [key]: headerValue },
      };
    }
  }
  return { strategy: "bearer", value: apiKey };
}

function getRemoteSecretFields(server: MCPSSEServer | MCPSHTTPServer) {
  let { auth, headers } = server;
  if (server.api_key) {
    const currentHeader = authorizationHeader(headers);
    if (!auth && currentHeader) {
      const [key, value] = currentHeader;
      headers = {
        ...headers,
        [key]: /^Bearer\s+/i.test(value)
          ? `Bearer ${server.api_key}`
          : server.api_key,
      };
    } else {
      auth = withApiKey(auth, server.api_key);
    }
  }
  return {
    ...(auth && { auth }),
    ...(headers && { headers }),
  };
}

/**
 * Generate a unique name for an MCP server, avoiding collisions with existing names.
 * Only adds a suffix if there's an actual collision.
 */
function getUniqueName(base: string, usedNames: Set<string>): string {
  if (!usedNames.has(base)) {
    return base;
  }
  let suffix = 1;
  while (usedNames.has(`${base}_${suffix}`)) {
    suffix += 1;
  }
  return `${base}_${suffix}`;
}

/**
 * Parse an SDK mcp_config value and convert it to the frontend MCPConfig
 * format used by UI components.
 *
 * Accepts both the SDK 1.31.x flat server map ({ shttp: {...} }) and the
 * legacy wrapped shape ({ mcpServers: { shttp: {...} } }) so previously
 * persisted settings keep loading. Preserves server names for round-trip
 * serialization.
 */
export function parseMcpConfig(value: unknown): MCPConfig {
  if (!value || typeof value !== "object") {
    return { ...EMPTY_MCP_CONFIG };
  }

  const obj = value as Record<string, unknown>;

  // Pass-through for the frontend's already-parsed grouped shape (used by
  // ``DEFAULT_AGENT_SETTINGS`` and any internal caller that hands us an
  // ``MCPConfig`` value). Detected by the three array-typed grouping keys.
  if (isParsedMcpConfigShape(obj)) {
    return {
      sse_servers: obj.sse_servers as (string | MCPSSEServer)[],
      stdio_servers: obj.stdio_servers as MCPStdioServer[],
      shttp_servers: obj.shttp_servers as (string | MCPSHTTPServer)[],
    };
  }

  const mcpServers = unwrapMcpServers(obj);

  const sseServers: (string | MCPSSEServer)[] = [];
  const stdioServers: MCPStdioServer[] = [];
  const shttpServers: (string | MCPSHTTPServer)[] = [];

  for (const [serverName, rawServerConfig] of Object.entries(mcpServers)) {
    if (
      !rawServerConfig ||
      typeof rawServerConfig !== "object" ||
      Array.isArray(rawServerConfig)
    ) {
      // eslint-disable-next-line no-continue
      continue;
    }

    const serverConfig = rawServerConfig as Record<string, unknown>;
    const url = serverConfig.url as string | undefined;

    if (url) {
      const transport = serverConfig.transport as string | undefined;
      const apiKey = apiKeyFromServerConfig(serverConfig);
      const auth = authCredential(serverConfig.auth);
      const headers = stringRecord(serverConfig.headers);

      if (transport === "sse") {
        const server: MCPSSEServer = {
          name: serverName,
          url,
        };
        if (apiKey) server.api_key = apiKey;
        if (auth) server.auth = auth;
        if (headers) server.headers = headers;
        sseServers.push(server);
      } else {
        const server: MCPSHTTPServer = {
          name: serverName,
          url,
        };
        if (apiKey) server.api_key = apiKey;
        if (auth) server.auth = auth;
        if (headers) server.headers = headers;
        if (serverConfig.timeout != null) {
          server.timeout = serverConfig.timeout as number;
        }
        shttpServers.push(server);
      }
    } else {
      const stdioServer: MCPStdioServer = {
        name: serverName,
        command: serverConfig.command as string,
      };
      if (serverConfig.args) {
        stdioServer.args = serverConfig.args as string[];
      }
      if (serverConfig.env) {
        stdioServer.env = serverConfig.env as Record<string, string>;
      }
      stdioServers.push(stdioServer);
    }
  }

  return {
    sse_servers: sseServers,
    stdio_servers: stdioServers,
    shttp_servers: shttpServers,
  };
}

/**
 * Convert the frontend MCPConfig format to the flat server map shape the
 * SDK 1.31.x settings model emits via ``model_dump`` (``agent_settings.mcp_config``
 * is now a ``dict[str, MCPServer]``, not a FastMCP ``MCPConfig``). The backend's
 * ``_normalize_mcp_config_field`` still accepts the legacy ``{ mcpServers: ... }``
 * wrapper for back-compat, so either shape round-trips; we emit the SDK-native
 * shape to match the wire format the frontend receives.
 *
 * Uses preserved names when available, only generates names for new servers.
 */
export function toSdkMcpConfig(
  config: MCPConfig,
): Record<string, SdkMcpServerConfig> | null {
  const mcpServers: Record<string, SdkMcpServerConfig> = {};
  const usedNames = new Set<string>();

  // SSE servers - use preserved name or generate
  for (const entry of config.sse_servers) {
    const server: SdkMcpServerConfig = {};
    if (typeof entry === "string") {
      server.url = entry;
    } else {
      server.url = entry.url;
      Object.assign(server, getRemoteSecretFields(entry));
    }
    server.transport = "sse";

    const baseName =
      typeof entry !== "string" && entry.name ? entry.name : "sse";
    const name = getUniqueName(baseName, usedNames);
    usedNames.add(name);
    mcpServers[name] = server;
  }

  // shttp servers - use preserved name or generate
  for (const entry of config.shttp_servers) {
    const server: SdkMcpServerConfig = {};
    if (typeof entry === "string") {
      server.url = entry;
    } else {
      server.url = entry.url;
      Object.assign(server, getRemoteSecretFields(entry));
      if (entry.timeout != null) server.timeout = entry.timeout;
    }

    const baseName =
      typeof entry !== "string" && entry.name ? entry.name : "shttp";
    const name = getUniqueName(baseName, usedNames);
    usedNames.add(name);
    mcpServers[name] = server;
  }

  // stdio servers - use preserved name or generate
  for (const entry of config.stdio_servers) {
    const server: SdkMcpServerConfig = {
      command: entry.command,
    };
    if (entry.args) server.args = entry.args;
    if (entry.env) server.env = entry.env;

    const baseName = entry.name || "stdio";
    const name = getUniqueName(baseName, usedNames);
    usedNames.add(name);
    mcpServers[name] = server;
  }

  return Object.keys(mcpServers).length > 0 ? mcpServers : null;
}
