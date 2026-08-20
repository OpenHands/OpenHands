import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Provider } from "#/types/settings";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { ExecutionStatus } from "#/types/agent-server/core";
import {
  resumeConversation,
  updateConversationExecutionStatusInCache,
  invalidateConversationQueries,
} from "./conversation-mutation-utils";

export const useUnifiedResumeConversation = () => {
  const queryClient = useQueryClient();
  const removeErrorMessage = useErrorMessageStore(
    (state) => state.removeErrorMessage,
  );

  return useMutation({
    mutationKey: ["start-conversation"],
    mutationFn: async (variables: {
      conversationId: string;
      providers?: Provider[];
    }) => resumeConversation(variables.conversationId),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousConversations = queryClient.getQueryData([
        "user",
        "conversations",
      ]);

      return { previousConversations };
    },
    onError: (_, __, context) => {
      if (context?.previousConversations) {
        queryClient.setQueryData(
          ["user", "conversations"],
          context.previousConversations,
        );
      }
    },
    onSettled: (_, __, variables) => {
      invalidateConversationQueries(queryClient, variables.conversationId);
    },
    onSuccess: (_, variables) => {
      removeErrorMessage();

      updateConversationExecutionStatusInCache(
        queryClient,
        variables.conversationId,
        ExecutionStatus.RUNNING,
      );
    },
  });
};
