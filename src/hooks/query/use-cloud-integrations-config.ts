import { useQuery } from "@tanstack/react-query";
import { fetchCloudIntegrationsConfig } from "#/api/cloud/settings-service.api";
import type { Backend } from "#/api/backend-registry/types";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * Query options for a cloud backend's web-client config, shared by the hook
 * and by non-React callers (the conversation launch) so both read one cache
 * entry per backend connection.
 */
export function cloudIntegrationsConfigQueryOptions(
  backend: Pick<Backend, "id" | "connectionRevision">,
) {
  return {
    queryKey: [
      "cloud-providers-configured",
      backend.id,
      backend.connectionRevision ?? 0,
    ],
    queryFn: fetchCloudIntegrationsConfig,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  } as const;
}

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
    ...cloudIntegrationsConfigQueryOptions(backend),
    enabled: enabled && backend.kind === "cloud",
  });
}
