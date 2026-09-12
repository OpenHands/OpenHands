import { useQuery } from "@tanstack/react-query";
import { getCloudOrganizationMember } from "#/api/cloud/organization-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * Resolve a single org member by `userId` on the active cloud backend via
 * `GET /api/organizations/{orgId}/members/{userId}` (currently read for the
 * member's email). Disabled on local backends, where there is no org to look
 * the user up in, and until both the active org and the user id are known.
 */
export function useCloudOrgMember(userId: string | undefined) {
  const { backend, orgId } = useActiveBackend();
  const enabled = backend.kind === "cloud" && !!orgId && !!userId;

  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  return useQuery({
    queryKey: [
      "cloud-org-member",
      backend.id,
      orgId,
      userId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => getCloudOrganizationMember(orgId!, userId!, backend),
    enabled,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });
}
