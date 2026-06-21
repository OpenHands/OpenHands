import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

/**
 * Pending invitations for the current org, including invite links.
 * The backing endpoint is gated on the invite permission (admins/owners),
 * so only enable this for callers that hold it.
 */
export const usePendingInvitations = (enabled: boolean) => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "pending-invitations", organizationId],
    queryFn: () =>
      organizationService.getPendingInvitations({ orgId: organizationId! }),
    enabled: enabled && !!organizationId,
  });
};
