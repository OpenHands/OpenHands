import { useQueries } from "@tanstack/react-query";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import {
  getCloudOrganizations,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";
import type { Backend } from "#/api/backend-registry/types";

/**
 * Fetch organizations for every registered cloud backend in parallel.
 *
 * Used by the BackendSelector to flatten each cloud backend into per-org
 * rows. Each query is keyed by the backend ID so React Query caches
 * responses independently and a switch (which clears the cache) refetches.
 */
export function useAllCloudOrganizations() {
  const { backends } = useActiveBackendContext();
  const cloudBackends = backends.filter((b) => b.kind === "cloud");

  const queries = useQueries({
    queries: cloudBackends.map((backend) => ({
      queryKey: ["cloud-organizations", backend.id],
      // Filter the user's full org membership down to the single org the
      // backend's API key is bound to. The cloud enforces one-key-one-org
      // server-side (HTTP 403 otherwise); without this filter the
      // selector would advertise orgs the key cannot use. Legacy keys
      // with no binding fall through unfiltered.
      queryFn: async () => {
        const orgs = await getCloudOrganizations(backend);
        if (backend.authMode === "cookie") return orgs;

        const key = await getCurrentCloudApiKey(backend);
        if (key.isLegacyKey || key.orgId === null) return orgs;
        return {
          ...orgs,
          items: orgs.items.filter((o) => o.id === key.orgId),
        };
      },
      staleTime: 1000 * 60 * 5,
      retry: false,
      meta: { disableToast: true },
    })),
  });

  // Map backend id -> result to make consumers easy to read.
  const byBackendId: Record<
    string,
    {
      backend: Backend;
      isLoading: boolean;
      orgs: { id: string; name: string; is_personal?: boolean }[];
      currentOrgId: string | null;
    }
  > = {};
  cloudBackends.forEach((backend, index) => {
    const q = queries[index];
    byBackendId[backend.id] = {
      backend,
      isLoading: q.isLoading,
      orgs: q.data?.items ?? [],
      currentOrgId: q.data?.currentOrgId ?? null,
    };
  });

  return byBackendId;
}
