import {
  MCPConfig,
  MCPSSEServer,
  MCPSHTTPServer,
  MCPStdioServer,
  SettingsValue,
} from "#/types/settings";
import { getMcpServerId, McpServerListEntry } from "#/utils/mcp-server-id";

/** System MCP server injected by the app server; not user-configurable. */
export const SYSTEM_MCP_SERVER_ID = "default";

const EMPTY_MCP_CONFIG: MCPConfig = {
  sse_servers: [],
  stdio_servers: [],
  shttp_servers: [],
};

type SdkMcpServerConfig = Record<string, SettingsValue>;
type SdkMcpConfig = { mcpServers: Record<string, SdkMcpServerConfig> };

export type McpServerType = "sse" | "stdio" | "shttp";

/** UI-level MCP server fields used for change detection and storage-key lookup. */
export type ComparableMcpServer = {
  type: McpServerType;
  name?: string;
  url?: string;
  api_key?: string;
  timeout?: number;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
};

function stringArraysEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function envRecordsEqual(
  a: Record<string, string>,
  b: Record<string, string>,
): boolean {
  const aKeys = Object.keys(a).sort();
  const bKeys = Object.keys(b).sort();
  return (
    stringArraysEqual(aKeys, bKeys) && aKeys.every((key) => a[key] === b[key])
  );
}

function apiKeysEqual(original?: string, updated?: string): boolean {
  const trimmed = updated?.trim();
  if (!trimmed) {
    // Blank on edit means "keep existing key".
    return true;
  }
  return original === trimmed;
}

/**
 * Returns true when two MCP server configs represent the same connection settings.
 * Ignores list ``id`` and treats a blank api_key as unchanged on edit.
 */
export function mcpServerConfigsEqual(
  original: ComparableMcpServer,
  updated: ComparableMcpServer,
): boolean {
  if (original.type !== updated.type) {
    return false;
  }

  if (original.type === "stdio") {
    return (
      original.name === updated.name &&
      original.command === updated.command &&
      stringArraysEqual(original.args ?? [], updated.args ?? []) &&
      envRecordsEqual(original.env ?? {}, updated.env ?? {})
    );
  }

  if (original.type === "shttp") {
    return (
      original.url === updated.url &&
      apiKeysEqual(original.api_key, updated.api_key) &&
      (original.timeout ?? undefined) === (updated.timeout ?? undefined)
    );
  }

  return (
    original.url === updated.url &&
    apiKeysEqual(original.api_key, updated.api_key)
  );
}

export function parseMcpEnvironmentVariables(
  envString: string,
): Record<string, string> {
  const env: Record<string, string> = {};
  const input = envString.trim();
  if (!input) return env;

  for (const line of input.split("\n")) {
    const trimmed = line.trim();
    const eq = trimmed.indexOf("=");
    const key = eq >= 0 ? trimmed.substring(0, eq).trim() : "";
    if (trimmed && eq !== -1 && key) {
      env[key] = trimmed.substring(eq + 1).trim();
    }
  }
  return env;
}

export function buildComparableMcpServerFromForm(
  formData: FormData,
  serverType: McpServerType,
): ComparableMcpServer {
  if (serverType === "sse" || serverType === "shttp") {
    const url = formData.get("url")?.toString().trim();
    const apiKey = formData.get("api_key")?.toString().trim();
    const timeoutStr = formData.get("timeout")?.toString().trim();

    const server: ComparableMcpServer = {
      type: serverType,
      url: url || undefined,
      ...(apiKey && { api_key: apiKey }),
    };

    if (serverType === "shttp" && timeoutStr) {
      const timeoutValue = parseInt(timeoutStr, 10);
      if (!Number.isNaN(timeoutValue)) {
        server.timeout = timeoutValue;
      }
    }

    return server;
  }

  const name = formData.get("name")?.toString().trim();
  const command = formData.get("command")?.toString().trim();
  const argsString = formData.get("args")?.toString().trim();
  const envString = formData.get("env")?.toString().trim();

  const args = argsString
    ? argsString
        .split("\n")
        .map((arg) => arg.trim())
        .filter(Boolean)
    : [];
  const env = parseMcpEnvironmentVariables(envString || "");

  return {
    type: "stdio",
    name: name || undefined,
    command: command || undefined,
    ...(args.length > 0 && { args }),
    ...(Object.keys(env).length > 0 && { env }),
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

type McpServerStorageKeyEntry = {
  type: McpServerType;
  index: number;
  key: string;
};

function listMcpServerStorageKeys(
  config: MCPConfig,
): McpServerStorageKeyEntry[] {
  const usedNames = new Set<string>();
  const entries: McpServerStorageKeyEntry[] = [];

  for (let index = 0; index < config.sse_servers.length; index += 1) {
    const entry = config.sse_servers[index];
    const baseName =
      typeof entry !== "string" && entry.name ? entry.name : "sse";
    const key = getUniqueName(baseName, usedNames);
    usedNames.add(key);
    entries.push({ type: "sse", index, key });
  }

  for (let index = 0; index < config.shttp_servers.length; index += 1) {
    const entry = config.shttp_servers[index];
    const baseName =
      typeof entry !== "string" && entry.name ? entry.name : "shttp";
    const key = getUniqueName(baseName, usedNames);
    usedNames.add(key);
    entries.push({ type: "shttp", index, key });
  }

  for (let index = 0; index < config.stdio_servers.length; index += 1) {
    const entry = config.stdio_servers[index];
    const baseName = entry.name || "stdio";
    const key = getUniqueName(baseName, usedNames);
    usedNames.add(key);
    entries.push({ type: "stdio", index, key });
  }

  return entries;
}

/** Resolve the backend ``mcpServers`` key for a server at the given list index. */
export function getMcpServerStorageKeyForEntry(
  config: MCPConfig,
  serverType: McpServerType,
  index: number,
): string | null {
  const entry = listMcpServerStorageKeys(config).find(
    (item) => item.type === serverType && item.index === index,
  );
  return entry?.key ?? null;
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
  return typeof auth === "string" && auth !== "oauth" ? auth : undefined;
}

/**
 * Serialize an API key as an ``Authorization`` bearer header. The SDK only
 * redacts/encrypts ``headers`` (and ``env``), not ``auth``, so a key written
 * to ``auth`` would persist in plaintext — write the header form instead.
 */
function getAuthorizationHeaders(apiKey: string | undefined) {
  if (!apiKey) return {};
  return {
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  };
}

/**
 * Parse an SDK mcp_config value ({ mcpServers: { ... } }) and convert it
 * to the frontend MCPConfig format used by UI components.
 * Preserves server names for round-trip serialization.
 */
export function parseMcpConfig(value: unknown): MCPConfig {
  if (!value || typeof value !== "object") {
    return { ...EMPTY_MCP_CONFIG };
  }

  const obj = value as Record<string, unknown>;

  if (
    !("mcpServers" in obj) ||
    !obj.mcpServers ||
    typeof obj.mcpServers !== "object"
  ) {
    return { ...EMPTY_MCP_CONFIG };
  }

  const sseServers: (string | MCPSSEServer)[] = [];
  const stdioServers: MCPStdioServer[] = [];
  const shttpServers: (string | MCPSHTTPServer)[] = [];

  const mcpServers = obj.mcpServers as Record<string, Record<string, unknown>>;

  for (const [serverName, serverConfig] of Object.entries(mcpServers)) {
    // eslint-disable-next-line no-continue
    if (!serverConfig || typeof serverConfig !== "object") continue;

    const url = serverConfig.url as string | undefined;

    if (url) {
      const transport = serverConfig.transport as string | undefined;
      const apiKey = apiKeyFromServerConfig(serverConfig);

      if (transport === "sse") {
        const server: MCPSSEServer = {
          name: serverName,
          url,
        };
        if (apiKey) server.api_key = apiKey;
        sseServers.push(server);
      } else {
        const server: MCPSHTTPServer = {
          name: serverName,
          url,
        };
        if (apiKey) server.api_key = apiKey;
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
 * Convert the frontend MCPConfig format back to the SDK { mcpServers: { ... } }
 * shape expected by agent_settings.mcp_config on the backend.
 * Uses preserved names when available, only generates names for new servers.
 */
export function toSdkMcpConfig(config: MCPConfig): SdkMcpConfig | null {
  const mcpServers: Record<string, SdkMcpServerConfig> = {};
  const usedNames = new Set<string>();

  // SSE servers - use preserved name or generate
  for (const entry of config.sse_servers) {
    const server: SdkMcpServerConfig = {};
    if (typeof entry === "string") {
      server.url = entry;
    } else {
      server.url = entry.url;
      Object.assign(server, getAuthorizationHeaders(entry.api_key));
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
      Object.assign(server, getAuthorizationHeaders(entry.api_key));
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

  return Object.keys(mcpServers).length > 0 ? { mcpServers } : null;
}

/** Return the storage key for a newly added server by diffing SDK configs. */
export function getAddedMcpServerStorageKey(
  before: MCPConfig,
  after: MCPConfig,
): string | null {
  const oldSdk = toSdkMcpConfig(before);
  const newSdk = toSdkMcpConfig(after);
  const oldKeys = new Set(Object.keys(oldSdk?.mcpServers ?? {}));
  return (
    Object.keys(newSdk?.mcpServers ?? {}).find((key) => !oldKeys.has(key)) ??
    null
  );
}

/** List user-configured MCP servers from settings, excluding the system default. */
export function listCustomMcpServers(
  config: MCPConfig | undefined,
): McpServerListEntry[] {
  if (!config) {
    return [];
  }

  const allServers: McpServerListEntry[] = [
    ...config.sse_servers.map((server, index) => ({
      id: `sse-${index}`,
      type: "sse" as const,
      name: typeof server === "object" ? server.name : undefined,
      url: typeof server === "string" ? server : server.url,
      api_key: typeof server === "object" ? server.api_key : undefined,
    })),
    ...config.stdio_servers.map((server, index) => ({
      id: `stdio-${index}`,
      type: "stdio" as const,
      name: server.name,
      command: server.command,
      args: server.args,
      env: server.env,
    })),
    ...config.shttp_servers.map((server, index) => ({
      id: `shttp-${index}`,
      type: "shttp" as const,
      name: typeof server === "object" ? server.name : undefined,
      url: typeof server === "string" ? server : server.url,
      api_key: typeof server === "object" ? server.api_key : undefined,
      timeout: typeof server === "object" ? server.timeout : undefined,
    })),
  ];

  return allServers.filter(
    (server) => getMcpServerId(server) !== SYSTEM_MCP_SERVER_ID,
  );
}
