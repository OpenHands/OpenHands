import { useInfiniteQuery } from "@tanstack/react-query";
import { sharedConversationService } from "#/api/shared-conversation-service.api";

export const useSharedConversationEvents = (conversationId?: string) =>
  useInfiniteQuery({
    queryKey: ["shared-conversation-events", conversationId],
    queryFn: ({ pageParam }) => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      return sharedConversationService.getSharedConversationEvents(
        conversationId,
        100,
        pageParam,
      );
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_page_id ?? undefined,
    enabled: !!conversationId,
    retry: false, // Don't retry for shared conversations
  });
