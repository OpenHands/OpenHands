import { useEffect } from "react";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useConversationRenderScope } from "#/contexts/conversation-render-scope";
import { useUserConversation } from "./use-user-conversation";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { isExecutionActive } from "#/utils/status";

interface UseActiveConversationOptions {
  /**
   * When false, fetch the conversation for the current navigation scope but
   * do not claim the process-wide `ConversationService` singleton. Popouts
   * use this so they don't overwrite the primary route's "current" conversation.
   * Defaults to the surrounding {@link useConversationRenderScope}'s `isPrimary`.
   */
  claimCurrentConversation?: boolean;
}

export const useActiveConversation = (
  options: UseActiveConversationOptions = {},
) => {
  const { isPrimary } = useConversationRenderScope();
  const { claimCurrentConversation = isPrimary } = options;
  // Optional: the chat input renders on the home page too (no conversation
  // route yet). The user-conversation query is gated on a real id below.
  const { conversationId } = useOptionalConversationId();

  // Task polling is handled by useTaskPolling hook
  const isTaskId = !!conversationId && conversationId.startsWith("task-");
  const actualConversationId =
    !conversationId || isTaskId ? null : conversationId;

  const userConversation = useUserConversation(
    actualConversationId,
    // Fast-poll (3 s) while: the sandbox URL is absent; the sandbox is PAUSED
    // (it keeps the stale conversation_url, so a missing-URL check alone misses
    // the wake-up); or the agent is executing but has no title yet (the title
    // lands asynchronously after conversation_url is already set).
    (query) => {
      const data = query.state.data;
      if (
        data &&
        (!data.conversation_url ||
          data.sandbox_status === "PAUSED" ||
          (!data.title && isExecutionActive(data.execution_status)))
      ) {
        return 3000;
      }
      return 30000;
    },
  );

  useEffect(() => {
    if (!claimCurrentConversation) {
      return;
    }
    const conversation = userConversation.data;
    ConversationService.setCurrentConversation(conversation || null);
  }, [
    claimCurrentConversation,
    conversationId,
    userConversation.isFetched,
    userConversation?.data?.execution_status,
  ]);
  return userConversation;
};
