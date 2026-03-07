import { useInfiniteQuery } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { ResultSet, Conversation } from "#/api/open-hands.types";
import { useIsAuthed } from "./use-is-authed";

/**
 * Guard against pagination loops where the backend returns a cursor we've
 * already seen. This prevents the UI from re-appending recent conversations
 * forever when users scroll to the end of history.
 */
export const getNextConversationPageParam = (
  lastPage: ResultSet<Conversation>,
  allPages: ResultSet<Conversation>[],
): string | undefined => {
  const nextPageId = lastPage.next_page_id;
  if (!nextPageId) {
    return undefined;
  }

  // Ignore the last page itself when checking for repeats.
  const previousPages = allPages.slice(0, -1);
  const alreadySeenCursor = previousPages.some(
    (page) => page.next_page_id === nextPageId,
  );

  return alreadySeenCursor ? undefined : nextPageId;
};

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
    getNextPageParam: getNextConversationPageParam,
    initialPageParam: undefined as string | undefined,
  });
};
