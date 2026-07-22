import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Provider } from "#/types/settings";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import {
  getRateLimitRetryDelayMs,
  isRateLimitError,
} from "#/utils/rate-limit-retry";
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
    // Mutation keys don't affect data cache - they only track mutation state.
    // This key is intentionally descriptive to distinguish from any legacy mutations.
    mutationKey: ["unified-resume-conversation-sandbox"],
    mutationFn: async (variables: {
      conversationId: string;
      providers?: Provider[];
    }) => resumeV1ConversationSandbox(variables.conversationId),
    retry: (failureCount, error) => isRateLimitError(error) && failureCount < 3,
    retryDelay: (failureCount, error) =>
      getRateLimitRetryDelayMs(failureCount, error),
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousConversations = queryClient.getQueryData([
        "user",
        "conversations",
      ]);
      const previousConversation =
        queryClient.getQueryData<V1AppConversation | null>([
          "user",
          "conversation",
          variables.conversationId,
        ]);

      queryClient.setQueryData<V1AppConversation | null>(
        ["user", "conversation", variables.conversationId],
        (oldData) =>
          oldData
            ? {
                ...oldData,
                sandbox_status: "STARTING",
                execution_status: null,
              }
            : oldData,
      );

      return { previousConversations, previousConversation };
    },
    onError: (_, __, context) => {
      if (context?.previousConversations) {
        queryClient.setQueryData(
          ["user", "conversations"],
          context.previousConversations,
        );
      }
      if (context?.previousConversation) {
        queryClient.setQueryData(
          ["user", "conversation", context.previousConversation.id],
          context.previousConversation,
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
