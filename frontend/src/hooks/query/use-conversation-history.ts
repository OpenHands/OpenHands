import { useQuery } from "@tanstack/react-query";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";

export const useConversationHistory = (conversationId?: string) => {
  const { data: conversation, isFetched: isConversationFetched } =
    useUserConversation(conversationId ?? null);

  const query = useQuery({
    queryKey: ["conversation-history", conversationId],
    enabled: !!conversationId && !!conversation,
    queryFn: async () => {
      if (!conversationId) return [];

      return EventService.searchEventsV1(conversationId);
    },
    staleTime: Infinity,
    gcTime: 30 * 60 * 1000, // 30 minutes — survive navigation away and back (AC5)
  });

  return {
    ...query,
    // Query is considered fetched when:
    // 1. Conversation data is fetched AND history query has run, OR
    // 2. Conversation doesn't exist (isConversationFetched && !conversation)
    isFetched: query.isFetched || (isConversationFetched && !conversation),
  };
};
