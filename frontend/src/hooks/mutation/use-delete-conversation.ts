import { useMutation, useQueryClient } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { clearConversationLocalStorage } from "#/utils/conversation-local-storage";
import {
  removeConversationsFromCache,
  restoreConversationsCache,
} from "./conversation-mutation-utils";

export const useDeleteConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: { conversationId: string }) =>
      V1ConversationService.deleteConversation(variables.conversationId),
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousData = removeConversationsFromCache(queryClient, [
        variables.conversationId,
      ]);
      return { previousData };
    },
    onSuccess: (_, variables) => {
      clearConversationLocalStorage(variables.conversationId);
    },
    onError: (_err, _variables, context) => {
      if (context?.previousData) {
        restoreConversationsCache(queryClient, context.previousData);
      }
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
