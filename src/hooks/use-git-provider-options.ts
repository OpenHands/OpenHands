import React from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudGitProviders } from "#/hooks/query/use-cloud-git-providers";
import { useCloudIntegrationsConfig } from "#/hooks/query/use-cloud-integrations-config";
import { useUserProviders } from "#/hooks/use-user-providers";
import type { GitProviderOption } from "#/utils/git-provider-option";
import { getProviderName } from "#/utils/utils";

/**
 * The git providers a repository picker offers.
 *
 * On a cloud instance with `enable_oauth_v2_login`, each linked connection that
 * does not need reconnecting is its own entry, labelled by its title, so two
 * connections of one type stay apart. Everywhere else the entries are the
 * connected provider types, with no id.
 */
export function useGitProviderOptions() {
  const { providers, isLoadingSettings } = useUserProviders();
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const { data: config, isPending: isConfigPending } =
    useCloudIntegrationsConfig();
  const isScoped = isCloud && config?.isOauthV2LoginEnabled === true;
  const { data: connections, isPending: isConnectionsPending } =
    useCloudGitProviders({ enabled: isScoped });

  const options = React.useMemo<GitProviderOption[]>(() => {
    if (isScoped) {
      return (connections ?? [])
        .filter((connection) => connection.connected)
        .filter((connection) => !connection.needsReconnect)
        .map((connection) => ({
          provider: connection.providerType,
          gitProviderId: connection.id,
          label: connection.title,
        }));
    }
    return providers.map((provider) => ({
      provider,
      gitProviderId: null,
      label: getProviderName(provider),
    }));
  }, [isScoped, connections, providers]);

  return {
    options,
    isScoped,
    isLoading:
      isLoadingSettings ||
      (isCloud && isConfigPending) ||
      (isScoped && isConnectionsPending),
  };
}
