import type { MCPServerConfig } from "#/types/mcp-server";

const sortedKeys = (record?: Record<string, string> | null): string[] =>
  Object.keys(record ?? {}).sort();

/**
 * Stable identity for a server's health entry.
 *
 * Built from the server's structural, non-secret fields. Names are included
 * because persisted configs are keyed by name (`toSdkMcpConfig` suffixes
 * duplicates `github` → `github_1`), which makes the key unique per stored
 * server and stable across list reordering — unlike the positional
 * `sse-0`/`stdio-1` ids from `flattenMcpConfig`. Secret VALUES are excluded:
 * the same credential legitimately appears as plaintext at install time,
 * `**********` in redacted settings, and ciphertext in test requests, so any
 * of them would make the key flap. A structural edit (URL, command, header
 * names, auth strategy, ...) therefore produces a new key, orphaning the old
 * health entry instead of misattributing it.
 */
export function getMcpServerHealthKey(server: MCPServerConfig): string {
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
