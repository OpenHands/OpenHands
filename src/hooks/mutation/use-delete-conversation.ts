import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { clearConversationLocalStorage } from "#/utils/conversation-local-storage";

export const useDeleteConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: { conversationId: string }) =>
      AgentServerConversationService.deleteConversation(
        variables.conversationId,
      ),
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
      // Mirror useCreateConversation: cloud surfaces in-flight
      // conversations via useStartTasks, so invalidate that key too so the
      // panel refreshes regardless of whether the deleted item was a ready
      // conversation or a still-provisioning start task.
      queryClient.invalidateQueries({ queryKey: ["start-tasks"] });
    },
  });
};
