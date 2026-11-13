import { useQuery } from "@tanstack/react-query";
import { fetchCloudIntegrationsConfig } from "#/api/cloud/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * The active cloud instance's web-client config: what its settings pages
 * offer and how it is deployed. Idle on local backends; read once per backend
 * connection and shared by every caller. `enabled: false` also skips the
 * request on a cloud backend (e.g. a personal workspace).
 */
export function useCloudIntegrationsConfig({
  enabled = true,
}: { enabled?: boolean } = {}) {
  const { backend } = useActiveBackend();

  return useQuery({
    queryKey: [
      "cloud-providers-configured",
      backend.id,
      backend.connectionRevision ?? 0,
    ],
    queryFn: fetchCloudIntegrationsConfig,
    enabled: enabled && backend.kind === "cloud",
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });
}
