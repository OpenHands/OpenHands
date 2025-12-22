import { useQuery } from "@tanstack/react-query";
import MicroagentManagementService from "#/api/microagent-management-service/microagent-management-service.api";

export const useMicroagentManagementConversations = (
  selectedRepository: string,
  pageId?: string,
  limit: number = 100,
  cacheDisabled: boolean = false,
) =>
  useQuery({
    queryKey: [
      "conversations",
      "microagent-management",
      pageId,
      limit,
      selectedRepository,
    ],
    queryFn: () =>
      MicroagentManagementService.getMicroagentManagementConversations(
        selectedRepository,
        pageId,
        limit,
      ),
    enabled: !!selectedRepository,
    staleTime: cacheDisabled ? 0 : 1000 * 60 * 5, // 5 minutes
    gcTime: cacheDisabled ? 0 : 1000 * 60 * 15, // 15 minutes
  });
