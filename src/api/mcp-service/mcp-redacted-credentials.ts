import SettingsService from "#/api/settings-service/settings-service.api";
import { isMcpAuthCredential } from "#/types/mcp-auth";
import type { MCPServerConfig } from "#/types/mcp-server";
import {
  hasRedactedMcpSecretLeaf,
  REDACTED_MCP_SECRET_VALUE,
} from "#/utils/mcp-config";

type StoredMcpServer = {
  url?: unknown;
  transport?: unknown;
  env?: unknown;
  auth?: unknown;
  headers?: unknown;
};

type StoredMcpConfig = Record<string, StoredMcpServer>;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);

const stringRecord = (value: unknown): Record<string, string> | undefined => {
  if (!isRecord(value)) return undefined;
  const entries = Object.entries(value).filter(
    (entry): entry is [string, string] => typeof entry[1] === "string",
  );
  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
};

const hasRedactedValue = (values: Record<string, string> | undefined) =>
  !!values &&
  Object.values(values).some((value) => value === REDACTED_MCP_SECRET_VALUE);

const remoteTransportMatches = (
  type: MCPServerConfig["type"],
  transport: unknown,
) => {
  if (type === "sse") return transport === "sse";
  if (type === "shttp") {
    return (
      transport === undefined ||
      transport === "http" ||
      transport === "shttp" ||
      transport === "streamable-http"
    );
  }
  return false;
};

// The editor assigns each stdio server an id of the form ``stdio-<i>`` matching
// its position in ``parseMcpConfig``'s stdio array. That position is stable
// across a rename because ``parseMcpConfig`` collects stdio entries (those
// without a ``url``) in ``Object.entries`` order, identical for the redacted
// and encrypted settings. Resolve the original stored entry by position so a
// renamed stdio server still finds its stored encrypted env, instead of
// looking it up by the new (no-longer-matching) display name.
const stdioIndexFromId = (id: string | undefined): number | undefined => {
  if (!id) return undefined;
  const match = /^stdio-(\d+)$/.exec(id);
  return match ? Number.parseInt(match[1], 10) : undefined;
};

const findStoredStdioByIndex = (
  id: string | undefined,
  storedServers: StoredMcpConfig,
): StoredMcpServer | undefined => {
  const index = stdioIndexFromId(id);
  if (index === undefined) return undefined;
  const stdioEntries = Object.entries(storedServers).filter(
    ([, stored]) => !stored.url,
  );
  return stdioEntries[index]?.[1];
};

const findStoredServer = (
  server: MCPServerConfig,
  storedServers: StoredMcpConfig,
): StoredMcpServer | undefined => {
  if (server.type === "stdio") {
    // Prefer the positional id: a rename changes the display name (the stored
    // dict key), and a rename onto an existing name would otherwise restore the
    // wrong server's secrets. Fall back to a name match only when no id-based
    // position is available (e.g. configs built without an editor-assigned id).
    return (
      findStoredStdioByIndex(server.id, storedServers) ??
      (server.name ? storedServers[server.name] : undefined)
    );
  }

  if (server.name && storedServers[server.name]) {
    return storedServers[server.name];
  }

  return Object.values(storedServers).find(
    (stored) =>
      stored.url === server.url &&
      remoteTransportMatches(server.type, stored.transport),
  );
};

async function fetchEncryptedStoredServer(
  server: MCPServerConfig,
): Promise<StoredMcpServer | undefined> {
  const response = await SettingsService.fetchSettingsFromApi("encrypted");
  const mcpConfig = response.agent_settings?.mcp_config;
  if (!isRecord(mcpConfig)) {
    return undefined;
  }
  return findStoredServer(server, mcpConfig as StoredMcpConfig);
}

/**
 * The MCP editor sees redacted settings (`**********`). When the user leaves
 * a secret unchanged, replace that placeholder with the stored encrypted
 * env/header/OAuth state value so tests and saves round-trip the real
 * credential without exposing plaintext in the browser.
 */
export async function substituteRedactedMcpCredentials(
  server: MCPServerConfig,
): Promise<MCPServerConfig> {
  const redactedStdioEnv =
    server.type === "stdio" && hasRedactedValue(server.env);
  const redactedRemoteAuth =
    (server.type === "sse" || server.type === "shttp") &&
    hasRedactedMcpSecretLeaf(server.auth);

  if (!redactedStdioEnv && !redactedRemoteAuth) {
    return server;
  }

  try {
    const stored = await fetchEncryptedStoredServer(server);
    if (!stored) return server;

    if (redactedStdioEnv) {
      const storedEnv = stringRecord(stored.env) ?? {};
      const env = Object.fromEntries(
        Object.entries(server.env ?? {}).map(([key, value]) => [
          key,
          value === REDACTED_MCP_SECRET_VALUE &&
          typeof storedEnv[key] === "string"
            ? storedEnv[key]
            : value,
        ]),
      );
      return { ...server, env };
    }

    if (!redactedRemoteAuth) return server;
    if (isMcpAuthCredential(stored.auth)) {
      return { ...server, auth: stored.auth };
    }
    return server;
  } catch {
    return server;
  }
}
