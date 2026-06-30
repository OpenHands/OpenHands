import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrgUserUsage = () => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "user-usage", organizationId],
    queryFn: () => organizationService.getUserUsageStats({ orgId: organizationId! }),
    enabled: !!organizationId,
  });
};
