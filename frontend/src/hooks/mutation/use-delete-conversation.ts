import { useMutation, useQueryClient } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { clearConversationLocalStorage } from "#/utils/conversation-local-storage";

export const useDeleteConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: { conversationId: string }) =>
      ConversationService.deleteUserConversation(variables.conversationId),
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousConversations = queryClient.getQueryData([
        "user",
        "conversations",
      ]);

      queryClient.setQueryData(
        ["user", "conversations"],
        (old: { conversation_id: string }[] | undefined) =>
          old?.filter(
            (conv) => conv.conversation_id !== variables.conversationId,
          ),
      );

      return { previousConversations };
    },

    onSuccess: (_, variables) => {
      clearConversationLocalStorage(variables.conversationId);
    },

    onError: (err, variables, context) => {
      if (context?.previousConversations) {
        queryClient.setQueryData(
          ["user", "conversations"],
          context.previousConversations,
        );
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
    },
  });
};
