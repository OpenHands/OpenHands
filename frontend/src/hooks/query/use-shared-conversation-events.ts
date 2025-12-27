import { useQuery } from "@tanstack/react-query";
import { sharedConversationService } from "#/api/shared-conversation-service.api";

export const useSharedConversationEvents = (conversationId?: string) =>
  useQuery({
    queryKey: ["shared-conversation-events", conversationId],
    queryFn: () => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      return sharedConversationService.getSharedConversationEvents(
        conversationId,
      );
    },
    enabled: !!conversationId,
    retry: false, // Don't retry for shared conversations
  });
