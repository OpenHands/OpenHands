import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Provider } from "#/types/settings";
import { useErrorMessageStore } from "#/stores/error-message-store";
import {
  getConversationVersionFromQueryCache,
  resumeV1ConversationSandbox,
  startV0Conversation,
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
      version?: "V0" | "V1";
    }) => {
      // Guard: If conversation is no longer in cache and no explicit version provided,
      // skip the mutation. This handles race conditions like org switching where cache
      // is cleared before the mutation executes.
      // We return undefined (not throw) to avoid triggering the global MutationCache.onError
      // handler which would display an error toast to the user.
      const cachedConversation = queryClient.getQueryData([
        "user",
        "conversation",
        variables.conversationId,
      ]);
      if (!cachedConversation && !variables.version) {
        return undefined;
      }

      // Use provided version or fallback to cache lookup
      const version =
        variables.version ||
        getConversationVersionFromQueryCache(
          queryClient,
          variables.conversationId,
        );

      if (version === "V1") {
        return resumeV1ConversationSandbox(variables.conversationId);
      }

      return startV0Conversation(variables.conversationId, variables.providers);
    },
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
