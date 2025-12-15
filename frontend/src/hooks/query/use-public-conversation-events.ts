import { useQuery } from "@tanstack/react-query";
import { publicConversationService } from "#/api/public-conversation-service.api";

export const usePublicConversationEvents = (conversationId?: string) =>
  useQuery({
    queryKey: ["public-conversation-events", conversationId],
    queryFn: () => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      return publicConversationService.getPublicConversationEvents(
        conversationId,
      );
    },
    enabled: !!conversationId,
    retry: false, // Don't retry for public conversations
  });
