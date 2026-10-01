import { useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchCloudProvidersConfigured } from "#/api/cloud/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useUserProviders } from "#/hooks/use-user-providers";
import type { Provider } from "#/types/settings";

/** Integration catalog entries a cloud instance can also connect natively. */
const NATIVE_PROVIDER_BY_ENTRY_ID: Record<string, Provider> = {
  github: "github",
  gitlab: "gitlab",
  bitbucket: "bitbucket",
};

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
 */
export function useNativeGitIntegrations() {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const { providers: connectedProviders } = useUserProviders();

  const { data: configuredProviders, isLoading } = useQuery({
    queryKey: [
      "cloud-providers-configured",
      backend.id,
      backend.connectionRevision ?? 0,
    ],
    queryFn: fetchCloudProvidersConfigured,
    enabled: isCloud,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });

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

  return { getNativeIntegration, isLoading };
}
