import { useQuery } from "@tanstack/react-query";
import { fetchCloudIntegrationsConfig } from "#/api/cloud/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const CLOUD_INTEGRATIONS_CONFIG_QUERY_KEY = [
  "cloud-providers-configured",
] as const;

/**
 * The cloud instance's web-client config (configured git providers, feature
 * flags). One cached request per backend connection, shared by every caller.
 * `enabled: false` skips the request (e.g. on a local backend).
 */
export function useCloudIntegrationsConfig({ enabled }: { enabled: boolean }) {
  const { backend } = useActiveBackend();
  return useQuery({
    queryKey: [
      ...CLOUD_INTEGRATIONS_CONFIG_QUERY_KEY,
      backend.id,
      backend.connectionRevision ?? 0,
    ],
    queryFn: fetchCloudIntegrationsConfig,
    enabled,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });
}
