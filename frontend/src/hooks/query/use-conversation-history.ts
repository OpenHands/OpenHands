import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";

export const useConversationHistory = (conversationId?: string) => {
  const { data: conversation } = useUserConversation(conversationId ?? null);
  const conversationVersion = conversation?.conversation_version;

  return useQuery({
    queryKey: ["conversation-history", conversationId, conversationVersion],
    enabled: !!conversationId && !!conversation,
    queryFn: async () => {
      if (!conversationId || !conversationVersion) return [];

      if (conversationVersion === "V1") {
        return EventService.searchEventsV1(conversationId);
      }

      return EventService.searchEventsV0(conversationId);
    },
    staleTime: Infinity,
    gcTime: 30 * 60 * 1000, // 30 minutes — survive navigation away and back (AC5)
  });
};
