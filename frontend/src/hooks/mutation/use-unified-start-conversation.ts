import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Provider } from "#/types/settings";
import { useErrorMessageStore } from "#/stores/error-message-store";
import {
  resumeV1ConversationSandbox,
  updateConversationSandboxStatusInCache,
  invalidateConversationQueries,
} from "./conversation-mutation-utils";

/**
 * Unified hook that automatically routes to the correct resume conversation sandbox implementation
 * based on the conversation version (V0 or V1).
 *
 * This hook checks the cached conversation data to determine the version, then calls
 * the appropriate API directly. Returns a single useMutation instance that all components share.
 *
 * Usage is the same as useStartConversation:
 * const { mutate: startConversation } = useUnifiedResumeConversationSandbox();
 * startConversation({ conversationId: "some-id", providers: [...] });
 */
export const useUnifiedResumeConversationSandbox = () => {
  const queryClient = useQueryClient();
  const removeErrorMessage = useErrorMessageStore(
    (state) => state.removeErrorMessage,
  );

  return useMutation({
    mutationKey: ["start-conversation"],
    mutationFn: async (variables: {
      conversationId: string;
      providers?: Provider[];
    }) => resumeV1ConversationSandbox(variables.conversationId),
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
      // Clear error messages when starting/resuming conversation
      removeErrorMessage();

      updateConversationSandboxStatusInCache(
        queryClient,
        variables.conversationId,
        "STARTING",
      );
    },
  });
};
