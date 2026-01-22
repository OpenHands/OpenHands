import { useInfiniteQuery } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useIsAuthed } from "./use-is-authed";

export const usePaginatedConversations = (limit: number = 20) => {
  const { data: userIsAuthenticated } = useIsAuthed();

  return useInfiniteQuery({
    queryKey: ["user", "conversations", "paginated", limit],
    queryFn: async ({ pageParam }) => {
      const result = await ConversationService.getUserConversations(
        limit,
        pageParam,
      );

      return result;
    },
    enabled: !!userIsAuthenticated,
    getNextPageParam: (lastPage) => lastPage.next_page_id,
    initialPageParam: undefined as string | undefined,
  });
};
