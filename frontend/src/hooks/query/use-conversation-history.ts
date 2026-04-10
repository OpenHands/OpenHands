import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";

export const useConversationHistory = (conversationId?: string) => {
  const { data: conversation } = useUserConversation(conversationId ?? null);

  return useQuery({
    queryKey: ["conversation-history", conversationId],
    enabled: !!conversationId && !!conversation,
    queryFn: async () => {
      if (!conversationId) return [];

      return EventService.searchEventsV1(conversationId);
    },
    staleTime: Infinity,
    gcTime: 30 * 60 * 1000, // 30 minutes — survive navigation away and back (AC5)
  });
};
