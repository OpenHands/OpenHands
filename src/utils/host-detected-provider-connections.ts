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
