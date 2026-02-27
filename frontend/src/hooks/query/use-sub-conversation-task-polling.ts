import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

/**
 * Hook that polls V1 sub-conversation start tasks.
 *
 * This hook:
 * - Polls the V1 start task API every 3 seconds until status is READY or ERROR
 * - Exposes task status and details for UI components to show loading states and errors
 *
 * Note: This hook does NOT invalidate the parent conversation cache. The component
 * that initiates the sub-conversation creation should handle cache invalidation
 * to ensure it only happens once.
 *
 * Use case:
 * - When creating a sub-conversation (e.g., plan mode), track the task status
 *   for UI loading states
 *
 * @param taskId - The task ID to poll (from createConversation response)
 * @param parentConversationId - The parent conversation ID (used to enable polling)
 */
export const useSubConversationTaskPolling = (
  taskId: string | null,
  parentConversationId: string | null,
) => {
  // Poll the task if we have both taskId and parentConversationId
  const taskQuery = useQuery({
    queryKey: ["sub-conversation-task", taskId],
    queryFn: async () => {
      if (!taskId) return null;
      return V1ConversationService.getStartTask(taskId);
    },
    enabled: !!taskId && !!parentConversationId,
    refetchInterval: (query) => {
      const task = query.state.data;
      if (!task) return false;

      // Stop polling if ready or error
      if (task.status === "READY" || task.status === "ERROR") {
        return false;
      }

      // Poll every 3 seconds while task is in progress
      return 3000;
    },
    retry: false,
  });

  return {
    task: taskQuery.data,
    taskStatus: taskQuery.data?.status,
    taskDetail: taskQuery.data?.detail,
    taskError: taskQuery.error,
    isLoadingTask: taskQuery.isLoading,
    subConversationId: taskQuery.data?.app_conversation_id,
  };
};
