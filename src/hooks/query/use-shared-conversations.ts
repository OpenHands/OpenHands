import { useInfiniteQuery } from "@tanstack/react-query";
import {
  searchCloudSharedConversations,
  type SharedConversationPage,
} from "#/api/cloud/shared-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  CONVERSATION_QUERY_KEYS,
  DEFAULT_CONVERSATION_PAGE_SIZE,
} from "./query-keys";

/**
 * The conversations shared with the active cloud org. The key sits under
 * `["user", "conversations"]`, so the mutations that change a conversation,
 * including the share toggle, refresh this list too.
 */
export const useSharedConversations = () => {
  const { backend, orgId } = useActiveBackend();

  return useInfiniteQuery({
    queryKey: CONVERSATION_QUERY_KEYS.shared(backend.id, orgId),
    queryFn: ({ pageParam }) =>
      searchCloudSharedConversations({
        limit: DEFAULT_CONVERSATION_PAGE_SIZE,
        pageId: pageParam,
      }),
    getNextPageParam: (lastPage: SharedConversationPage) =>
      lastPage.next_page_id,
    initialPageParam: undefined as string | undefined,
  });
};
