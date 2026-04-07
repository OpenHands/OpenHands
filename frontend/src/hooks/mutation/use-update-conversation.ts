import { useMutation, useQueryClient } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

export const useUpdateConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: { conversationId: string; newTitle: string }) =>
      V1ConversationService.updateConversationTitle(
        variables.conversationId,
        variables.newTitle,
      ),
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousConversations = queryClient.getQueryData([
        "user",
        "conversations",
      ]);

      queryClient.setQueryData(
        ["user", "conversations"],
        (old: { id: string; title: string }[] | undefined) =>
          old?.map((conv) =>
            conv.id === variables.conversationId
              ? { ...conv, title: variables.newTitle }
              : conv,
          ),
      );

      // Also optimistically update the active conversation query
      queryClient.setQueryData(
        ["user", "conversation", variables.conversationId],
        (old: { title: string } | undefined) =>
          old ? { ...old, title: variables.newTitle } : old,
      );

      return { previousConversations };
    },
    onError: (err, variables, context) => {
      if (context?.previousConversations) {
        queryClient.setQueryData(
          ["user", "conversations"],
          context.previousConversations,
        );
      }
    },
    onSettled: (data, error, variables) => {
      // Invalidate and refetch the conversation list to show the updated title
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
