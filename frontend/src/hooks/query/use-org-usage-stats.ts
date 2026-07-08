import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

interface UseOrgUsageStatsParams {
  days?: number;
  timeWindow?: string;
}

export const useOrgUsageStats = ({
  days,
  timeWindow,
}: UseOrgUsageStatsParams = {}) => {
  const { organizationId } = useSelectedOrganizationId();
  const queryWindow = timeWindow ?? days ?? "default";

  return useQuery({
    queryKey: [
      "organizations",
      "usage-stats",
      organizationId,
      queryWindow,
    ],
    queryFn: () =>
      organizationService.getUsageStats({
        orgId: organizationId!,
        days,
        timeWindow,
      }),
    enabled: !!organizationId,
  });
};
