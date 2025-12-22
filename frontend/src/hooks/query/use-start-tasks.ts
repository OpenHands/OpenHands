import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useSettings } from "#/hooks/query/use-settings";

/**
 * Hook to fetch in-progress V1 conversation start tasks
 *
 * Use case: Show tasks that are provisioning sandboxes, cloning repos, etc.
 * These are conversations that started but haven't reached READY or ERROR status yet.
 *
 * Note: Filters out READY and ERROR status tasks client-side since backend doesn't support status filtering.
 *
 * @param limit Maximum number of tasks to return (max 100)
 * @returns Query result with array of in-progress start tasks
 */
export const useStartTasks = (limit = 10) => {
  const { data: settings } = useSettings();
  const isV1Enabled = settings?.v1_enabled;

  return useQuery({
    queryKey: ["start-tasks", "search", limit],
    queryFn: () => V1ConversationService.searchStartTasks(limit),
    enabled: isV1Enabled,
    select: (tasks) =>
      tasks.filter(
        (task) => task.status !== "READY" && task.status !== "ERROR",
      ),
  });
};
