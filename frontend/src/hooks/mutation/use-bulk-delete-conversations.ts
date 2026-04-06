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
    mutationFn: async (variables: { conversationIds: string[] }) => {
      await Promise.all(
        variables.conversationIds.map((id) =>
          ConversationService.deleteUserConversation(id),
        ),
      );
    },
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousData = removeConversationsFromCache(
        queryClient,
        variables.conversationIds,
      );
      return { previousData };
    },
    onSuccess: (_, variables) => {
      variables.conversationIds.forEach(clearConversationLocalStorage);
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
