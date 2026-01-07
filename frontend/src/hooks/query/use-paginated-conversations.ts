import { useEffect } from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { cleanupOrphanedConversationLocalStorage } from "#/utils/conversation-local-storage";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useIsAuthed } from "./use-is-authed";

export const usePaginatedConversations = (limit: number = 20) => {
  const { data: userIsAuthenticated } = useIsAuthed();
  const queryClient = useQueryClient();

  const query = useInfiniteQuery({
    queryKey: ["user", "conversations", "paginated", limit],
    queryFn: async ({ pageParam }) => {
      const result = await ConversationService.getUserConversations(
        limit,
        pageParam,
      );

      // Optimistically populate individual conversation caches
      result.results.forEach((conversation) => {
        queryClient.setQueryData(
          ["user", "conversation", conversation.conversation_id],
          conversation,
        );
      });

      return result;
    },
    enabled: !!userIsAuthenticated,
    getNextPageParam: (lastPage) => lastPage.next_page_id,
    initialPageParam: undefined as string | undefined,
  });

  useEffect(() => {
    if (!query.data) return;

    const conversationIds = query.data.pages
      .flatMap((page) => page.results)
      .map((conversation) => conversation.conversation_id);

    cleanupOrphanedConversationLocalStorage(conversationIds);
  }, [query.data]);

  return query;
};
