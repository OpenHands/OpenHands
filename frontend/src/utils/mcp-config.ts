import {
  MCPConfig,
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

type SdkMcpServerConfig = Record<string, SettingsValue>;
type SdkMcpConfig = { mcpServers: Record<string, SdkMcpServerConfig> };

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
      const auth = serverConfig.auth as string | undefined;
      const apiKey =
        typeof auth === "string" && auth !== "oauth" ? auth : undefined;

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
      if (entry.api_key) server.auth = entry.api_key;
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
      if (entry.api_key) server.auth = entry.api_key;
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
