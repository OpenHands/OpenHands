import { useQuery } from "@tanstack/react-query";
import { publicConversationService } from "#/api/public-conversation-service.api";

export const usePublicConversation = (conversationId?: string) =>
  useQuery({
    queryKey: ["public-conversation", conversationId],
    queryFn: () => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      return publicConversationService.getPublicConversation(conversationId);
    },
    enabled: !!conversationId,
    retry: false, // Don't retry for public conversations
  });
