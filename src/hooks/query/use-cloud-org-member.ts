import { useQueries, useQuery } from "@tanstack/react-query";
import { getCloudOrganizationMember } from "#/api/cloud/organization-service.api";
import type { Backend } from "#/api/backend-registry/types";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * One query definition for an org member, shared by the single and the batch
 * lookup so both read and fill the same cache entry.
 */
function cloudOrgMemberQuery(
  backend: Backend,
  orgId: string | null | undefined,
  userId: string | undefined,
) {
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  return {
    queryKey: [
      "cloud-org-member",
      backend.id,
      orgId,
      userId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => getCloudOrganizationMember(orgId!, userId!, backend),
    enabled: backend.kind === "cloud" && !!orgId && !!userId,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  };
}

/**
 * Resolve a single org member by `userId` on the active cloud backend via
 * `GET /api/organizations/{orgId}/members/{userId}` (currently read for the
 * member's email). Disabled on local backends, where there is no org to look
 * the user up in, and until both the active org and the user id are known.
 */
export function useCloudOrgMember(userId: string | undefined) {
  const { backend, orgId } = useActiveBackend();

  return useQuery(cloudOrgMemberQuery(backend, orgId, userId));
}

/** The org members for `userIds`, by id, once each lookup has settled. */
export function useCloudOrgMembers(
  userIds: readonly string[],
  { enabled = true } = {},
) {
  const { backend, orgId } = useActiveBackend();

  return useQueries({
    queries: userIds.map((userId) => {
      const query = cloudOrgMemberQuery(backend, orgId, userId);
      return { ...query, enabled: enabled && query.enabled };
    }),
    combine: (results) =>
      new Map(
        userIds.map((userId, index) => [
          userId,
          {
            email: results[index]?.data?.email,
            isLoading: !!results[index]?.isLoading,
          },
        ]),
      ),
  });
}
