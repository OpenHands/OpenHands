import React from "react";
import type { Provider } from "#/types/settings";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudGitProviders } from "#/hooks/query/use-cloud-git-providers";
import { useCloudIntegrationsConfig } from "#/hooks/query/use-cloud-integrations-config";
import { useSettings } from "#/hooks/query/use-settings";

/**
 * A mapping from provider type to its host URL. The host is null for hosted
 * services (github.com, gitlab.com) or when the server does not report one.
 */
export type ProviderHostsMap = Partial<Record<Provider, string | null>>;

/**
 * Build a provider-to-host mapping from CloudGitProvider connections.
 * When multiple connections of the same provider type exist, the first
 * non-null host wins.
 */
export function buildProviderHostsFromConnections(
  connections: Array<{
    providerType: Provider;
    connected: boolean;
    host: string | null;
  }>,
): ProviderHostsMap {
  const map: ProviderHostsMap = {};
  for (const connection of connections) {
    if (!connection.connected) continue;
    const provider = connection.providerType;
    // Only set if not already set (first wins)
    if (!(provider in map)) {
      map[provider] = connection.host;
    } else if (map[provider] === null && connection.host !== null) {
      // Upgrade null to a non-null host if found later
      map[provider] = connection.host;
    }
  }
  return map;
}

/**
 * The host URL for each connected git provider.
 *
 * On a cloud instance with `enable_oauth_v2_login`, hosts come from the
 * CloudGitProvider connections. When multiple connections of the same type
 * exist, the first non-null host wins. Everywhere else hosts come from
 * `settings.provider_tokens_set`.
 *
 * The returned mapping is suitable for passing to `providerHostKnown()` and
 * `automationRepositoryAddress()`.
 */
export function useProviderHosts(): {
  hosts: ProviderHostsMap;
  isLoading: boolean;
} {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";

  const { data: config, isPending: isConfigPending } =
    useCloudIntegrationsConfig();
  const isScoped = isCloud && config?.isOauthV2LoginEnabled === true;

  const { data: connections, isPending: isConnectionsPending } =
    useCloudGitProviders({ enabled: isScoped });

  const { data: settings, isLoading: isSettingsLoading } = useSettings();

  const hosts = React.useMemo<ProviderHostsMap>(() => {
    if (isScoped && connections) {
      return buildProviderHostsFromConnections(connections);
    }
    // Fall back to settings-based hosts (local backend or non-OAuth v2)
    return settings?.provider_tokens_set ?? {};
  }, [isScoped, connections, settings?.provider_tokens_set]);

  return {
    hosts,
    isLoading:
      isSettingsLoading ||
      (isCloud && isConfigPending) ||
      (isScoped && isConnectionsPending),
  };
}
