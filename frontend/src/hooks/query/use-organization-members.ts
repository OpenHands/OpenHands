import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

interface UseOrganizationMembersParams {
  page?: number;
  limit?: number;
  email?: string;
}

export const useOrganizationMembers = ({
  page = 1,
  limit = 10,
  email,
}: UseOrganizationMembersParams = {}) => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "members", organizationId, page, limit, email],
    queryFn: () =>
      organizationService.getOrganizationMembers({
        orgId: organizationId!,
        page,
        limit,
        email: email || undefined,
      }),
    enabled: !!organizationId,
    placeholderData: keepPreviousData,
  });
};
