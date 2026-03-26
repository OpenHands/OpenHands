import { useEffect, useRef } from "react";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useUserConversation } from "./use-user-conversation";
import ConversationService from "#/api/conversation-service/conversation-service.api";

export const useActiveConversation = () => {
  const { conversationId } = useConversationId();

  // Don't poll if this is a task ID (format: "task-{uuid}")
  // Task polling is handled by useTaskPolling hook
  const isTaskId = conversationId.startsWith("task-");
  const actualConversationId = isTaskId ? null : conversationId;

  // Track when this conversation was first loaded to poll faster initially.
  // This helps pick up metadata (selected_repository, title, etc.) that may
  // still be committing when the task transitions to READY.
  const loadedAtRef = useRef<number>(0);
  const prevCidRef = useRef(actualConversationId);
  if (prevCidRef.current !== actualConversationId) {
    prevCidRef.current = actualConversationId;
    loadedAtRef.current = actualConversationId ? Date.now() : 0;
  }

  const userConversation = useUserConversation(
    actualConversationId,
    (query) => {
      if (query.state.data?.status === "STARTING") {
        return 3000; // 3 seconds
      }
      // Poll frequently for the first 15 seconds after navigating to a new
      // conversation to ensure we pick up metadata that may still be committing
      if (loadedAtRef.current && Date.now() - loadedAtRef.current < 15_000) {
        return 3000; // 3 seconds
      }
      // TODO: Return conversation title as a WS event to avoid polling
      // This was changed from 5 minutes to 30 seconds to poll for updated conversation title after an auto update
      return 30000; // 30 seconds
    },
  );

  useEffect(() => {
    const conversation = userConversation.data;
    ConversationService.setCurrentConversation(conversation || null);
  }, [
    conversationId,
    userConversation.isFetched,
    userConversation?.data?.status,
  ]);
  return userConversation;
};
