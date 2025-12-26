import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrganizationMembers = () => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "members", organizationId],
    queryFn: () =>
      organizationService.getOrganizationMembers({ orgId: organizationId! }),
    enabled: !!organizationId,
  });
};
