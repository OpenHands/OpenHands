import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

interface UseOrgConversationsParams {
  page?: number;
  perPage?: number;
  search?: string;
  sortBy?: string;
  sortOrder?: string;
  executionStatus?: string;
  sandboxStatus?: string;
  timeWindow?: string;
  includeSubConversations?: boolean;
}

export const useOrgConversations = ({
  page = 1,
  perPage = 20,
  search = "",
  sortBy = "updated_at",
  sortOrder = "desc",
  executionStatus = "",
  sandboxStatus = "",
  timeWindow = "",
  includeSubConversations = false,
}: UseOrgConversationsParams = {}) => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: [
      "organizations",
      "conversations",
      organizationId,
      page,
      perPage,
      search,
      sortBy,
      sortOrder,
      executionStatus,
      sandboxStatus,
      timeWindow,
      includeSubConversations,
    ],
    queryFn: () =>
      organizationService.getConversations({
        orgId: organizationId!,
        page,
        perPage,
        search: search || undefined,
        sortBy,
        sortOrder,
        executionStatus: executionStatus || undefined,
        sandboxStatus: sandboxStatus || undefined,
        timeWindow: timeWindow || undefined,
        includeSubConversations,
      }),
    enabled: !!organizationId,
    placeholderData: keepPreviousData,
  });
};
