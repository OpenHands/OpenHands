import { useQuery } from "@tanstack/react-query";
import { sharedConversationService } from "#/api/shared-conversation-service.api";

export const useSharedConversation = (conversationId?: string) =>
  useQuery({
    queryKey: ["shared-conversation", conversationId],
    queryFn: () => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      return sharedConversationService.getSharedConversation(conversationId);
    },
    enabled: !!conversationId,
    retry: false, // Don't retry for shared conversations
  });
