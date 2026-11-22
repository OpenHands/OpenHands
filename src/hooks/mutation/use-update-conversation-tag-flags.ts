import {
  useMutation,
  useQueryClient,
  type QueryKey,
} from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type {
  AppConversation,
  AppConversationPage,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import { withConversationTagFlags } from "#/utils/conversation-tag-flags";

interface InfiniteConversationsData {
  pages: AppConversationPage[];
  pageParams: unknown[];
}

function applyTagFlags<
  T extends { id: string; tags?: Record<string, string> | null },
>(
  conversation: T,
  conversationId: string,
  changes: Record<string, boolean>,
): T {
  if (conversation.id !== conversationId) return conversation;
  return {
    ...conversation,
    tags: withConversationTagFlags(conversation.tags, changes),
  };
}

/**
 * Sets/clears server-side conversation tag flags (`pinned`, `archived`, …)
 * with optimistic updates across every cache shape that can hold the
 * conversation: the single-conversation query, the flat conversation list,
 * and the paginated sidebar list. Rolls back on error and refetches on
 * settle so the sidebar converges on the server state.
 */
export const useUpdateConversationTagFlags = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: {
      conversationId: string;
      currentTags: Record<string, string> | null | undefined;
      changes: Record<string, boolean>;
    }) =>
      AgentServerConversationService.updateConversationTagFlags(
        variables.conversationId,
        variables.currentTags,
        variables.changes,
      ),
    onMutate: async (variables) => {
      const conversationPrefix: QueryKey = [
        "user",
        "conversation",
        variables.conversationId,
      ];
      const listPrefix: QueryKey = ["user", "conversations"];
      await queryClient.cancelQueries({ queryKey: conversationPrefix });
      await queryClient.cancelQueries({ queryKey: listPrefix });

      const previousConversationEntries = queryClient.getQueriesData<
        AppConversation | null | undefined
      >({ queryKey: conversationPrefix });
      const previousListEntries = queryClient.getQueriesData<unknown>({
        queryKey: listPrefix,
      });

      queryClient.setQueriesData<AppConversation | null | undefined>(
        { queryKey: conversationPrefix },
        (old) =>
          old
            ? applyTagFlags(old, variables.conversationId, variables.changes)
            : old,
      );

      queryClient.setQueriesData<unknown>(
        { queryKey: listPrefix },
        (old: unknown) => {
          if (Array.isArray(old)) {
            return old.map((conversation) =>
              conversation && typeof conversation === "object"
                ? applyTagFlags(
                    conversation as AppConversation,
                    variables.conversationId,
                    variables.changes,
                  )
                : conversation,
            );
          }
          if (
            old &&
            typeof old === "object" &&
            Array.isArray((old as InfiniteConversationsData).pages)
          ) {
            const data = old as InfiniteConversationsData;
            return {
              ...data,
              pages: data.pages.map((page) => ({
                ...page,
                items: page.items.map((conversation) =>
                  applyTagFlags(
                    conversation,
                    variables.conversationId,
                    variables.changes,
                  ),
                ),
              })),
            };
          }
          return old;
        },
      );

      return { previousConversationEntries, previousListEntries };
    },
    onError: (_err, _variables, context) => {
      context?.previousConversationEntries.forEach(([queryKey, data]) => {
        queryClient.setQueryData(queryKey, data);
      });
      context?.previousListEntries.forEach(([queryKey, data]) => {
        queryClient.setQueryData(queryKey, data);
      });
    },
    onSettled: (_data, _error, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });
    },
  });
};
