import { useQueries } from "@tanstack/react-query";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import {
  useActiveBackend,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import { useAllCloudOrganizations } from "./use-cloud-organizations";

/**
 * Resolve the current user's `user_id` per cloud backend with one
 * `/api/organizations/{orgId}/me` call per backend (NOT one per org).
 *
 * The cloud contract: `/me` returns `{ org_id, user_id, … }`. `user_id`
 * is identical regardless of which org you ask, so we make a single
 * call per backend.
 *
 * Path-param rule: when `backend.id === active.backend.id` and
 * `active.orgId` is set, the call uses **that** orgId — i.e. `/me`
 * always tracks the currently selected environment for the active
 * backend. When `includeInactive` is enabled, inactive backends use their
 * first org as a sentinel just to obtain `user_id`. This matches the
 * requirement that `/me` reflect the selected org for the active environment,
 * while still supporting personal-workspace labels when the backend picker or
 * management UI explicitly loads non-active backends.
 *
 * The query key includes `active.orgId`, so picking a different org
 * via `setActive` re-keys this query and refetches `/me` with the new
 * active orgId.
 */
export interface UseCloudCurrentUserIdOptions {
  /** Resolve identities for non-active Cloud backends as well. */
  includeInactive?: boolean;
}

export function useCloudCurrentUserId(
  options: UseCloudCurrentUserIdOptions = {},
): Record<string, { isLoading: boolean; userId: string | null }> {
  const { includeInactive = false } = options;
  const { backends } = useActiveBackendContext();
  const active = useActiveBackend();
  const cloudOrgs = useAllCloudOrganizations({ includeInactive });

  const targets: {
    backendId: string;
    connectionRevision: number;
    orgIdForMe: string;
  }[] = [];
  for (const backend of backends) {
    if (
      backend.kind === "cloud" &&
      (includeInactive || backend.id === active.backend.id)
    ) {
      const entry = cloudOrgs[backend.id];
      // Prefer the active org when this backend IS the active one and
      // an org has been selected; otherwise fall back to the first org
      // we know about for that backend.
      const preferredOrgId =
        backend.id === active.backend.id && active.orgId
          ? active.orgId
          : (entry?.orgs[0]?.id ?? null);
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
