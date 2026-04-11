import { useInfiniteQuery, useIsMutating } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useIsAuthed } from "./use-is-authed";
import { V1AppConversationPage } from "#/api/conversation-service/v1-conversation-service.types";

export const usePaginatedConversations = (limit: number = 20) => {
  const { data: userIsAuthenticated } = useIsAuthed();
  const isBulkDeleting =
    useIsMutating({ mutationKey: ["bulk-delete-conversations"] }) > 0;

  return useInfiniteQuery({
    queryKey: ["user", "conversations", "paginated", limit],
    queryFn: async ({ pageParam }) => {
      const result = await V1ConversationService.searchConversations(
        limit,
        pageParam,
      );

      return result;
    },
    enabled: !!userIsAuthenticated && !isBulkDeleting,
    getNextPageParam: (lastPage: V1AppConversationPage) =>
      lastPage.next_page_id,
    initialPageParam: undefined as string | undefined,
  });
};
