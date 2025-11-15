import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrganization = () => {
  const { orgId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", orgId],
    queryFn: () => organizationService.getOrganization({ orgId: orgId! }),
    enabled: !!orgId,
  });
};
