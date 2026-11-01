import { CONVERSATION_LAUNCH_PREFIX } from "#/stores/conversation-launch-store";
import { useEffect } from "react";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useUserConversation } from "./use-user-conversation";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { isExecutionErrored } from "#/utils/status";
import { isPlaceholderConversationTitle } from "#/utils/conversation-title";

const STARTUP_TITLE_POLL_WINDOW_MS = 60_000;

export const useActiveConversation = () => {
  // Optional: the chat input renders on the home page too (no conversation
  // route yet). The user-conversation query is gated on a real id below.
  const { conversationId } = useOptionalConversationId();

  // Task polling is handled by useTaskPolling hook
  const isTaskId =
    !!conversationId &&
    (conversationId.startsWith("task-") ||
      conversationId.startsWith(CONVERSATION_LAUNCH_PREFIX));
  const actualConversationId =
    !conversationId || isTaskId ? null : conversationId;

  const userConversation = useUserConversation(
    actualConversationId,
    // Keep startup/resume responsive; title-only polling stops after one minute.
    (query) => {
      const data = query.state.data;
      const age = data ? Date.now() - Date.parse(data.created_at) : Infinity;
      const generatingTitle =
        data?.title_generation?.status === "pending" &&
        Date.now() < Date.parse(data.title_generation.deadline_at);
      if (generatingTitle) return 500;
      const awaitingTitle =
        !data?.title_generation &&
        data &&
        isPlaceholderConversationTitle(data.title, data.id) &&
        !isExecutionErrored(data.execution_status) &&
        age >= 0 &&
        age < STARTUP_TITLE_POLL_WINDOW_MS;
      if (
        data &&
        (!data.conversation_url ||
          data.sandbox_status === "PAUSED" ||
          awaitingTitle)
      ) {
        return 3000;
      }
      return 30000;
    },
  );

  useEffect(() => {
    const conversation = userConversation.data;
    ConversationService.setCurrentConversation(conversation || null);
  }, [
    conversationId,
    userConversation.isFetched,
    userConversation?.data?.execution_status,
  ]);
  return userConversation;
};
