import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

interface UseOrganizationMembersCountParams {
  email?: string;
}

export const useOrganizationMembersCount = ({
  email,
}: UseOrganizationMembersCountParams = {}) => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "members", "count", organizationId, email],
    queryFn: () =>
      organizationService.getOrganizationMembersCount({
        orgId: organizationId!,
        email: email || undefined,
      }),
    enabled: !!organizationId,
    placeholderData: keepPreviousData,
  });
};
