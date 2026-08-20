import { useQuery } from "@tanstack/react-query";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * Hook to fetch in-progress conversation start tasks
 *
 * Use case: Show tasks that are provisioning sandboxes, cloning repos, etc.
 * These are conversations that started but haven't reached READY or ERROR status yet.
 *
 * Note: Filters out READY and ERROR status tasks client-side since backend doesn't support status filtering.
 *
 * @param limit Maximum number of tasks to return (max 100)
 * @returns Query result with array of in-progress start tasks
 */
export const useStartTasks = (limit = 10) =>
  useQuery({
    queryKey: ["start-tasks", "search", limit],
    queryFn: (): AppConversationStartTask[] => [],
    select: (tasks) =>
      tasks.filter(
        (task) => task.status !== "READY" && task.status !== "ERROR",
      ),
  });
