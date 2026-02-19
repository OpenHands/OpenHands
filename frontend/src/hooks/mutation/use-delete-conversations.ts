import { useMutation, useQueryClient } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { clearConversationLocalStorage } from "#/utils/conversation-local-storage";

export const useDeleteConversations = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (variables: { conversationIds: string[] }) => {
      await Promise.allSettled(
        variables.conversationIds.map((id) =>
          ConversationService.deleteUserConversation(id),
        ),
      );
    },
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousConversations = queryClient.getQueryData([
        "user",
        "conversations",
      ]);

      const idsToDelete = new Set(variables.conversationIds);
      queryClient.setQueryData(
        ["user", "conversations"],
        (old: { conversation_id: string }[] | undefined) =>
          old?.filter((conv) => !idsToDelete.has(conv.conversation_id)),
      );

      return { previousConversations };
    },

    onSuccess: (_, variables) => {
      for (const id of variables.conversationIds) {
        clearConversationLocalStorage(id);
      }
    },

    onError: (_err, _variables, context) => {
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
