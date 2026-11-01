import { useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchCloudIntegrationsConfig,
  type CloudIntegrationsConfig,
} from "#/api/cloud/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useUserProviders } from "#/hooks/use-user-providers";
import type { Provider } from "#/types/settings";

/** Integration catalog entries a cloud instance can also connect natively. */
const NATIVE_PROVIDER_BY_ENTRY_ID: Record<string, Provider> = {
  github: "github",
  gitlab: "gitlab",
  bitbucket: "bitbucket",
};

/** The built-in integrations an instance's config says it enables. */
function getBuiltInIntegrations(
  config: CloudIntegrationsConfig,
): ReadonlySet<string> {
  const ids = new Set<string>(config.providersConfigured);
  if (config.isJiraEnabled) ids.add("jira");
  if (config.isJiraDcEnabled) ids.add("jira_dc");
  if (config.isLinearEnabled) ids.add("linear");
  if (config.isSlackEnabled) ids.add("slack");
  return ids;
}

export interface NativeGitIntegration {
  provider: Provider;
  isConnected: boolean;
}

/** Whether a catalog entry is one a cloud instance may connect natively. */
export function isNativeGitCandidate(entryId: string): boolean {
  return Object.hasOwn(NATIVE_PROVIDER_BY_ENTRY_ID, entryId);
}

/**
 * Resolves whether an integration catalog entry can use the active cloud
 * instance's native git integration (Settings > Integrations) instead of an
 * MCP server. `getNativeIntegration` returns null — meaning MCP is the only
 * option — on local backends, for non-git entries, and for providers the
 * instance has not enabled. While `isLoading`, the instance's providers are
 * not known yet and null means nothing: wait rather than decide on it.
 * `isJiraEnabled`, `isLinearEnabled` and `isSlackEnabled` are whether the
 * instance offers the built-in Jira Cloud, Linear and Slack integrations
 * there; they are not git connections, so they never satisfy an integration
 * requirement.
 * `deploymentMode` is the instance's reported deployment mode (`cloud` or
 * `self_hosted`), null on local backends and on instances that report none.
 * `builtInIntegrations` names every integration the instance builds in, by
 * provider name (`github`, `bitbucket_data_center`, `jira_dc`, ...), or is null
 * while that is unknown: on local backends, while loading, and when the
 * instance config could not be read.
 */
export function useNativeGitIntegrations() {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const { providers: connectedProviders } = useUserProviders();

  const { data: config, isLoading } = useQuery({
    queryKey: [
      "cloud-providers-configured",
      backend.id,
      backend.connectionRevision ?? 0,
    ],
    queryFn: fetchCloudIntegrationsConfig,
    enabled: isCloud,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });
  const configuredProviders = config?.providersConfigured;

  const getNativeIntegration = useCallback(
    (entryId: string): NativeGitIntegration | null => {
      const provider = NATIVE_PROVIDER_BY_ENTRY_ID[entryId];
      if (!isCloud || !provider || !configuredProviders?.includes(provider)) {
        return null;
      }
      return { provider, isConnected: connectedProviders.includes(provider) };
    },
    [isCloud, configuredProviders, connectedProviders],
  );

  const builtInIntegrations = useMemo(
    () => (isCloud && config ? getBuiltInIntegrations(config) : null),
    [isCloud, config],
  );

  return {
    getNativeIntegration,
    builtInIntegrations,
    isJiraEnabled: isCloud && config?.isJiraEnabled === true,
    isLinearEnabled: isCloud && config?.isLinearEnabled === true,
    isSlackEnabled: isCloud && config?.isSlackEnabled === true,
    deploymentMode: isCloud ? (config?.deploymentMode ?? null) : null,
    isLoading,
  };
}
