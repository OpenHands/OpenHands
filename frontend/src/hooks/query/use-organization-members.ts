import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrganizationMembers = () => {
  const { orgId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "members", orgId],
    queryFn: () =>
      organizationService.getOrganizationMembers({ orgId: orgId! }),
    enabled: !!orgId,
  });
};
