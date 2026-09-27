import { useQueries } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import axios from "axios";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import {
  getCloudOrganizations,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";
import type { Backend } from "#/api/backend-registry/types";

function isAuthorizationError(error: unknown): boolean {
  const status =
    error instanceof HttpError
      ? error.status
      : axios.isAxiosError(error)
        ? error.response?.status
        : undefined;
  return status === 401 || status === 403;
}

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
      queryKey: [
        "cloud-organizations",
        backend.id,
        backend.connectionRevision ?? 0,
      ],
      // Filter the user's full org membership down to the single org the
      // backend's API key is bound to. The cloud enforces one-key-one-org
      // server-side (HTTP 403 otherwise); without this filter the
      // selector would advertise orgs the key cannot use. Legacy keys
      // with no binding fall through unfiltered.
      queryFn: async () => {
        const orgs = await getCloudOrganizations(backend);
        // Drop orgs the cloud marks invisible (`is_visible === false`, set
        // from its `HIDE_PERSONAL_WORKSPACES` policy). Every consumer —
        // the selector rows, the "Cloud Settings" deep link, the manage
        // modal and the synced-settings badge — reads orgs from here, so
        // filtering once keeps them from disagreeing about which orgs
        // exist. Only an explicit `false` hides an org: an app-server that
        // predates the field omits it, and those orgs stay visible.
        const visible = orgs.items.filter((o) => o.is_visible !== false);
        if (backend.authMode === "cookie") {
          return { ...orgs, items: visible };
        }

        const key = await getCurrentCloudApiKey(backend);
        if (key.isLegacyKey || key.orgId === null) {
          return { ...orgs, items: visible };
        }
        return {
          ...orgs,
          items: visible.filter((o) => o.id === key.orgId),
        };
      },
      staleTime: 1000 * 60 * 5,
      retry: (failureCount: number, error: unknown) =>
        failureCount < 2 && !isAuthorizationError(error),
      // Mounting workspace consumers must not restart a failed startup query.
      retryOnMount: false,
      meta: { disableToast: true },
    })),
  });

  // Map backend id -> result to make consumers easy to read.
  const byBackendId: Record<
    string,
    {
      backend: Backend;
      isLoading: boolean;
      isSuccess: boolean;
      isFetching: boolean;
      isError: boolean;
      isAuthorizationError: boolean;
      hasData: boolean;
      refetch: () => unknown;
      orgs: { id: string; name: string; is_personal?: boolean }[];
      currentOrgId: string | null;
    }
  > = {};
  cloudBackends.forEach((backend, index) => {
    const q = queries[index];
    byBackendId[backend.id] = {
      backend,
      isLoading: q.isLoading,
      isSuccess: q.isSuccess,
      isFetching: q.isFetching,
      isError: q.isError,
      isAuthorizationError: isAuthorizationError(q.error),
      hasData: q.data !== undefined,
      refetch: q.refetch,
      orgs: q.data?.items ?? [],
      currentOrgId: q.data?.currentOrgId ?? null,
    };
  });

  return byBackendId;
}
