import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";

export const useConversationHistory = (conversationId?: string) => {
  const { data: conversation } = useUserConversation(conversationId ?? null);

  return useQuery({
    queryKey: ["conversation-history", conversationId, conversation],
    enabled: !!conversationId && !!conversation,
    queryFn: async () => {
      if (!conversationId || !conversation) return [];

      if (conversation.conversation_version === "V1") {
        return EventService.searchEventsV1(conversationId);
      }

      return EventService.searchEventsV0(conversationId);
    },
    staleTime: 30_000,
  });
};
