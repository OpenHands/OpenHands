import { useMutation, useQueryClient } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

export const useUpdateConversationPublicFlag = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: { conversationId: string; isPublic: boolean }) =>
      V1ConversationService.updateConversationPublicFlag(
        variables.conversationId,
        variables.isPublic,
      ),
    onMutate: async (variables) => {
      await queryClient.cancelQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });
      const previousConversation = queryClient.getQueryData([
        "user",
        "conversation",
        variables.conversationId,
      ]);

      // Optimistically update the conversation's public flag
      queryClient.setQueryData(
        ["user", "conversation", variables.conversationId],
        (old: { public?: boolean } | undefined) =>
          old ? { ...old, public: variables.isPublic } : old,
      );

      return { previousConversation };
    },
    onError: (err, variables, context) => {
      if (context?.previousConversation) {
        queryClient.setQueryData(
          ["user", "conversation", variables.conversationId],
          context.previousConversation,
        );
      }
    },
    onSettled: (data, error, variables) => {
      // Invalidate and refetch the conversation to show the updated public flag
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });

      // Also invalidate the conversation list in case it affects the list view
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
