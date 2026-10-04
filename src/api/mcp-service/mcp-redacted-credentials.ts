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

/**
 * Recursively replace only the redacted display placeholders inside
 * ``submitted`` with the corresponding stored (encrypted) leaf. Fresh
 * non-redacted edits and untouched structure are preserved verbatim, so a
 * probe request exercises the server exactly as it will be persisted.
 */
function substituteRedactedLeaves<T>(submitted: T, stored: unknown): T {
  if (!isRecord(submitted)) return submitted;

  const storedRecord = isRecord(stored) ? stored : {};
  const merged: Record<string, unknown> = { ...submitted };
  for (const [key, value] of Object.entries(submitted)) {
    if (value === REDACTED_MCP_SECRET_VALUE) {
      const storedValue = storedRecord[key];
      if (
        typeof storedValue === "string" &&
        storedValue !== REDACTED_MCP_SECRET_VALUE
      ) {
        merged[key] = storedValue;
      }
    } else if (isRecord(value)) {
      merged[key] = substituteRedactedLeaves(value, storedRecord[key]);
    }
  }
  return merged as T;
}

const findStoredServer = (
  server: MCPServerConfig,
  storedServers: StoredMcpConfig,
): StoredMcpServer | undefined => storedServers[server.id];

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
 * env/header/OAuth state value so a connectivity test can exercise the real
 * credential without exposing plaintext in the browser. Persistence never
 * calls this helper; sparse settings patches omit unchanged secrets.
 */
export async function substituteRedactedMcpCredentials(
  server: MCPServerConfig,
): Promise<MCPServerConfig> {
  const redactedStdioEnv =
    server.type === "stdio" && hasRedactedValue(server.env);
  const redactedRemoteAuth =
    (server.type === "sse" || server.type === "shttp") &&
    hasRedactedMcpSecretLeaf(server.auth);
  const redactedRemoteHeaders =
    (server.type === "sse" || server.type === "shttp") &&
    hasRedactedValue(server.headers);

  if (!redactedStdioEnv && !redactedRemoteAuth && !redactedRemoteHeaders) {
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

    const nextServer = { ...server };
    if (redactedRemoteAuth && isMcpAuthCredential(stored.auth)) {
      nextServer.auth = substituteRedactedLeaves(server.auth, stored.auth);
    }
    if (redactedRemoteHeaders) {
      const storedHeaders = stringRecord(stored.headers) ?? {};
      nextServer.headers = Object.fromEntries(
        Object.entries(server.headers ?? {}).map(([key, value]) => [
          key,
          value === REDACTED_MCP_SECRET_VALUE &&
          typeof storedHeaders[key] === "string"
            ? storedHeaders[key]
            : value,
        ]),
      );
    }
    return nextServer;
  } catch {
    return server;
  }
}
