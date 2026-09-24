import {
  useMutation,
  useQueryClient,
  type QueryKey,
} from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

export const useUpdateConversationTags = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: {
      conversationId: string;
      tags: Record<string, string>;
    }) =>
      AgentServerConversationService.updateConversationTags(
        variables.conversationId,
        variables.tags,
      ),
    onMutate: async (variables) => {
      // The active conversation is cached under a prefix-extended key
      // (`["user", "conversation", id, backendId, orgId]`) by
      // `useUserConversation`. Cancel/snapshot/update via the prefix so the
      // optimistic change actually reaches the rendered query, not a stale
      // 3-element key that no observer ever reads.
      const prefix: QueryKey = [
        "user",
        "conversation",
        variables.conversationId,
      ];
      await queryClient.cancelQueries({ queryKey: prefix });
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });

      const previousEntries = queryClient.getQueriesData<
        AppConversation | null | undefined
      >({ queryKey: prefix });
      const previousConversations = queryClient.getQueryData([
        "user",
        "conversations",
      ]);

      queryClient.setQueriesData<AppConversation | null | undefined>(
        { queryKey: prefix },
        (old) => (old ? { ...old, tags: variables.tags } : old),
      );

      queryClient.setQueryData(
        ["user", "conversations"],
        (
          old:
            | { id: string; tags?: Record<string, string> | null }[]
            | undefined,
        ) =>
          old?.map((conv) =>
            conv.id === variables.conversationId
              ? { ...conv, tags: variables.tags }
              : conv,
          ),
      );

      return { previousEntries, previousConversations };
    },
    onError: (err, variables, context) => {
      context?.previousEntries.forEach(([key, data]) => {
        queryClient.setQueryData(key, data);
      });
      if (context?.previousConversations) {
        queryClient.setQueryData(
          ["user", "conversations"],
          context.previousConversations,
        );
      }
    },
    onSettled: (data, error, variables) => {
      // Invalidate and refetch the conversation list to show the updated tags
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });

      // Also invalidate the specific conversation query
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });
    },
  });
};
