import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrgUserUsage = ({
  limit,
  offset,
}: {
  limit?: number;
  offset?: number;
} = {}) => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "user-usage", organizationId, limit, offset],
    queryFn: () =>
      organizationService.getUserUsageStats({
        orgId: organizationId!,
        limit,
        offset,
      }),
    enabled: !!organizationId,
  });
};
