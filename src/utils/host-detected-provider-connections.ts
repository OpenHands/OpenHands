import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";

export const HOST_DETECTED_CONNECTION_PREFIX = "host:";

const CLI_ONLY_PROVIDERS = new Set(["cursor-cli", "opencode"]);

export function isHostDetectedConnection(
  connection: Pick<ProviderConnection, "id">,
): boolean {
  return connection.id.startsWith(HOST_DETECTED_CONNECTION_PREFIX);
}

/**
 * Stored API-key connections stay as-is. Host CLI / ChatGPT logins are
 * prepended unless a stored row already represents that CLI.
 */
export function mergeHostDetectedConnections(
  stored: ProviderConnection[],
  detected: ProviderConnection[],
): ProviderConnection[] {
  const storedIds = new Set(stored.map((connection) => connection.id));
  const storedCliProviders = new Set(
    stored
      .filter((connection) => CLI_ONLY_PROVIDERS.has(connection.provider))
      .map((connection) => connection.provider),
  );

  const extras = detected.filter((connection) => {
    if (storedIds.has(connection.id)) return false;
    if (storedCliProviders.has(connection.provider)) return false;
    return true;
  });

  return [...extras, ...stored];
}

export function buildHostDetectedConnection(
  id: string,
  displayName: string,
  provider: string,
): ProviderConnection {
  return {
    id: `${HOST_DETECTED_CONNECTION_PREFIX}${id}`,
    display_name: displayName,
    provider,
    base_url: null,
    created_at: 0,
    updated_at: 0,
    api_key_set: true,
  };
}

export const HOST_CHATGPT_CONNECTION_ID = `${HOST_DETECTED_CONNECTION_PREFIX}chatgpt`;
export const HOST_CLAUDE_CONNECTION_ID = `${HOST_DETECTED_CONNECTION_PREFIX}anthropic`;
export const HOST_CURSOR_CONNECTION_ID = `${HOST_DETECTED_CONNECTION_PREFIX}cursor-cli`;
export const HOST_OPENCODE_CONNECTION_ID = `${HOST_DETECTED_CONNECTION_PREFIX}opencode`;

export const CURSOR_CLI_ACP_COMMAND = ["agent", "acp"] as const;
export const OPENCODE_ACP_COMMAND = ["opencode", "acp"] as const;
export const CURSOR_CLI_AUTO_AGENT_PROFILE_NAME = "cursor-cli";
export const OPENCODE_AUTO_AGENT_PROFILE_NAME = "opencode";

export type PreferredHostLaunch =
  | { kind: "chatgpt" }
  | {
      kind: "acp";
      acpServer: "claude-code" | "custom";
      profileName: string;
      command?: readonly string[];
    };

/**
 * Choose a conversation launch path from host-detected subscriptions.
 * Preference: Claude Code (built-in ACP), ChatGPT (OpenHands LLM), then
 * Cursor CLI / OpenCode via their native ACP commands.
 */
export function pickPreferredHostLaunch(
  connections: ReadonlyArray<Pick<ProviderConnection, "id" | "provider">>,
): PreferredHostLaunch | null {
  const ids = new Set(connections.map((connection) => connection.id));
  const providers = new Set(
    connections.map((connection) => connection.provider),
  );
  if (ids.has(HOST_CLAUDE_CONNECTION_ID)) {
    return {
      kind: "acp",
      acpServer: "claude-code",
      profileName: "claude-code",
    };
  }
  if (ids.has(HOST_CHATGPT_CONNECTION_ID)) {
    return { kind: "chatgpt" };
  }
  if (ids.has(HOST_CURSOR_CONNECTION_ID) || providers.has("cursor-cli")) {
    return {
      kind: "acp",
      acpServer: "custom",
      profileName: CURSOR_CLI_AUTO_AGENT_PROFILE_NAME,
      command: CURSOR_CLI_ACP_COMMAND,
    };
  }
  if (ids.has(HOST_OPENCODE_CONNECTION_ID) || providers.has("opencode")) {
    return {
      kind: "acp",
      acpServer: "custom",
      profileName: OPENCODE_AUTO_AGENT_PROFILE_NAME,
      command: OPENCODE_ACP_COMMAND,
    };
  }
  return null;
}
