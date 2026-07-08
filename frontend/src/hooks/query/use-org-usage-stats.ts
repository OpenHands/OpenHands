import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

interface UseOrgUsageStatsParams {
  days?: number;
}

export const useOrgUsageStats = ({ days = 7 }: UseOrgUsageStatsParams = {}) => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "usage-stats", organizationId, days],
    queryFn: () =>
      organizationService.getUsageStats({ orgId: organizationId!, days }),
    enabled: !!organizationId,
  });
};
