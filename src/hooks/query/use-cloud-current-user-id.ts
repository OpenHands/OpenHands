import { useQueries } from "@tanstack/react-query";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import {
  useActiveBackend,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import { useAllCloudOrganizations } from "./use-cloud-organizations";

/** Resolve one authorized membership per backend to identify personal workspaces. */
export function useCloudCurrentUserId(): Record<
  string,
  { isLoading: boolean; userId: string | null }
> {
  const { backends } = useActiveBackendContext();
  const active = useActiveBackend();
  const cloudOrgs = useAllCloudOrganizations();

  const targets: {
    backendId: string;
    connectionRevision: number;
    orgIdForMe: string;
  }[] = [];
  for (const backend of backends) {
    if (backend.kind === "cloud") {
      const entry = cloudOrgs[backend.id];
      if (!entry?.hasData || entry.isAuthorizationError) continue;
      const isActiveBackend = backend.id === active.backend.id;
      if (
        isActiveBackend &&
        active.orgId &&
        !entry.orgs.some((org) => org.id === active.orgId)
      ) {
        // Wait for selection repair so the request's X-Org-Id matches its path.
        continue;
      }
      const preferredOrgId =
        isActiveBackend && active.orgId
          ? active.orgId
          : (entry.orgs[0]?.id ?? null);
      if (preferredOrgId) {
        targets.push({
          backendId: backend.id,
          connectionRevision: backend.connectionRevision ?? 0,
          orgIdForMe: preferredOrgId,
        });
      }
    }
  }

  const results = useQueries({
    queries: targets.map(({ backendId, connectionRevision, orgIdForMe }) => {
      const backend = backends.find((b) => b.id === backendId);
      return {
        // `orgIdForMe` is included in the query key so re-resolving the
        // active org also re-keys this query → React Query refetches
        // automatically without relying on explicit invalidation.
        queryKey: [
          "cloud-current-user",
          backendId,
          orgIdForMe,
          connectionRevision,
        ] as const,
        queryFn: async () => {
          if (!backend) return { orgId: orgIdForMe, userId: "" };
          return getCloudOrganizationMe(orgIdForMe, backend);
        },
        enabled: !!backend,
        staleTime: 1000 * 60 * 5,
        retry: false,
        meta: { disableToast: true },
      };
    }),
  });

  const out: Record<string, { isLoading: boolean; userId: string | null }> = {};
  targets.forEach((target, index) => {
    const q = results[index];
    out[target.backendId] = {
      isLoading: q.isLoading,
      userId: q.data?.userId ?? null,
    };
  });
  return out;
}
