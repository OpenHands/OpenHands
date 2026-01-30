import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { setConversationState } from "#/utils/conversation-local-storage";
import { useConversationStore } from "#/stores/conversation-store";

/**
 * Hook that polls V1 sub-conversation start tasks and invalidates parent conversation cache when ready.
 *
 * This hook:
 * - Polls the V1 start task API every 3 seconds until status is READY or ERROR
 * - Automatically invalidates the parent conversation cache when the task becomes READY
 * - Exposes task status and details for UI components to show loading states and errors
 *
 * Use case:
 * - When creating a sub-conversation (e.g., plan mode), track the task and refresh parent conversation
 *   data once the sub-conversation is ready
 *
 * @param taskId - The task ID to poll (from createConversation response)
 * @param parentConversationId - The parent conversation ID to invalidate when ready
 */
export const useSubConversationTaskPolling = (
  taskId: string | null,
  parentConversationId: string | null,
) => {
  const queryClient = useQueryClient();
  const { setSubConversationTaskId } = useConversationStore();

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

  // Invalidate parent conversation cache when task is ready and clear localStorage
  useEffect(() => {
    const task = taskQuery.data;
    if (!parentConversationId) return;

    if (task?.status === "READY" && task.app_conversation_id) {
      // Invalidate the parent conversation to refetch with updated sub_conversation_ids
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", parentConversationId],
      });
      // Clear the task ID from localStorage and store since task completed successfully
      setConversationState(parentConversationId, {
        subConversationTaskId: null,
      });
      setSubConversationTaskId(null);
    } else if (task?.status === "ERROR") {
      // Clear the task ID from localStorage and store on error so user can retry
      setConversationState(parentConversationId, {
        subConversationTaskId: null,
      });
      setSubConversationTaskId(null);
    }
  }, [
    taskQuery.data,
    parentConversationId,
    queryClient,
    setSubConversationTaskId,
  ]);

  return {
    task: taskQuery.data,
    taskStatus: taskQuery.data?.status,
    taskDetail: taskQuery.data?.detail,
    taskError: taskQuery.error,
    isLoadingTask: taskQuery.isLoading,
    subConversationId: taskQuery.data?.app_conversation_id,
  };
};
