import type { MCPServerConfig } from "#/types/mcp-server";

const sortedKeys = (record?: Record<string, string> | null): string[] =>
  Object.keys(record ?? {}).sort();

/**
 * Identity scope for an MCP server's health entry. The same server
 * structure on two backends is two independent installs with two
 * independent credentials and reachability, so their health verdicts
 * must not share storage. `backendId` distinguishes backends; the
 * optional `connectionRevision` rotates whenever the backend's
 * credentials change, returning the health to "unchecked" without the
 * caller having to clear the entry explicitly.
 */
export interface McpServerHealthScope {
  backendId: string;
  connectionRevision?: number;
}

/**
 * Stable identity for a server's health entry.
 *
 * Built from the server's structural, non-secret fields. Names are included
 * because persisted configs are keyed by name, which makes the key unique per
 * stored server and stable across list reordering. Secret VALUES are excluded:
 * the same credential legitimately appears as plaintext at install time,
 * `**********` in redacted settings, and ciphertext in test requests, so any
 * of them would make the key flap. A structural edit (URL, command, header
 * names, auth strategy, ...) therefore produces a new key, orphaning the old
 * health entry instead of misattributing it.
 */
export function getMcpServerHealthKey(
  scope: McpServerHealthScope,
  server: MCPServerConfig,
): string {
  const serverPart = serverKeyPart(server);
  const scopePart = `b=${scope.backendId}|r=${scope.connectionRevision ?? 0}`;
  return `${scopePart}|${serverPart}`;
}

function serverKeyPart(server: MCPServerConfig): string {
  if (server.type === "stdio") {
    return JSON.stringify({
      type: server.type,
      name: server.name ?? "",
      command: server.command ?? "",
      args: server.args ?? [],
      envKeys: sortedKeys(server.env),
    });
  }
  return JSON.stringify({
    type: server.type,
    name: server.name ?? "",
    url: server.url ?? "",
    headerKeys: sortedKeys(server.headers),
    authStrategy: server.auth?.strategy ?? "",
    authHeaderName:
      server.auth?.strategy === "api_key"
        ? (server.auth.header_name ?? "")
        : "",
    authHeaderKeys:
      server.auth?.strategy === "header" ? sortedKeys(server.auth.headers) : [],
  });
}
