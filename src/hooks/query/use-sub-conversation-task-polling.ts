import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useConversationStore } from "#/stores/conversation-store";
import { setConversationState } from "#/utils/conversation-local-storage";

/**
 * Hook that polls V1 sub-conversation start tasks.
 *
 * This hook:
 * - Polls the V1 start task API every 3 seconds until status is READY or ERROR
 * - Exposes task status and details for UI components to show loading states and errors
 * - Clears a terminal ERROR ownership claim that produced no planner conversation, so
 *   an explicit Plan retry can create a replacement (#19380)
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
  const setSubConversationTaskId = useConversationStore(
    (state) => state.setSubConversationTaskId,
  );

  // Poll the task if we have both taskId and parentConversationId
  const taskQuery = useQuery({
    queryKey: ["sub-conversation-task", taskId],
    queryFn: async () => {
      if (!taskId) return null;
      return AgentServerConversationService.getStartTask(taskId);
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

  // A Cloud planner start that ends in ERROR with no conversation id still left
  // its task id in the store and localStorage, so hasPlanner stayed true and
  // Plan retries sent no create. Drop that ownership so an explicit retry can
  // start a replacement; in-flight / READY / existing planners are unchanged.
  useEffect(() => {
    const task = taskQuery.data;
    if (!taskId || !parentConversationId || !task) return;
    if (task.status !== "ERROR") return;
    if (task.app_conversation_id) return;

    setSubConversationTaskId(null);
    setConversationState(parentConversationId, {
      subConversationTaskId: null,
    });
  }, [taskQuery.data, taskId, parentConversationId, setSubConversationTaskId]);

  return {
    task: taskQuery.data,
    taskStatus: taskQuery.data?.status,
    taskDetail: taskQuery.data?.detail,
    taskError: taskQuery.error,
    isLoadingTask: taskQuery.isLoading,
    subConversationId: taskQuery.data?.app_conversation_id,
  };
};
