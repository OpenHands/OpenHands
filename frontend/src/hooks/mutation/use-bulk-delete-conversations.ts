import { useMutation, useQueryClient } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { clearConversationLocalStorage } from "#/utils/conversation-local-storage";
import {
  removeConversationsFromCache,
  restoreConversationsCache,
} from "./conversation-mutation-utils";

export const useBulkDeleteConversations = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: { conversationIds: string[] }) =>
      ConversationService.bulkDeleteConversations(variables.conversationIds),
    onMutate: async (variables) => {
      // Cancel any in-flight fetches and save snapshot for rollback
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousData = removeConversationsFromCache(
        queryClient,
        variables.conversationIds,
      );
      return { previousData };
    },
    onSuccess: (data) => {
      data.succeeded.forEach(clearConversationLocalStorage);
    },
    onError: (_err, _variables, context) => {
      if (context?.previousData) {
        restoreConversationsCache(queryClient, context.previousData);
      }
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
