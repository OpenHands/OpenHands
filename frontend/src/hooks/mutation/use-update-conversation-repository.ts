import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { I18nKey } from "#/i18n/declaration";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import { Provider } from "#/types/settings";

interface UpdateRepositoryVariables {
  conversationId: string;
  repository: string | null;
  branch?: string | null;
  gitProvider?: Provider | null;
}

export const useUpdateConversationRepository = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: (variables: UpdateRepositoryVariables) =>
      V1ConversationService.updateConversationRepository(
        variables.conversationId,
        variables.repository,
        variables.branch,
        variables.gitProvider,
      ),
    onMutate: async (variables) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });

      // Snapshot the previous value
      const previousConversation = queryClient.getQueryData([
        "user",
        "conversation",
        variables.conversationId,
      ]);

      // Optimistically update the conversation
      queryClient.setQueryData(
        ["user", "conversation", variables.conversationId],
        (old: unknown) =>
          old && typeof old === "object"
            ? {
                ...old,
                selected_repository: variables.repository,
                selected_branch: variables.branch,
                git_provider: variables.gitProvider,
              }
            : old,
      );

      return { previousConversation };
    },
    onError: (err, variables, context) => {
      // Rollback on error
      if (context?.previousConversation) {
        queryClient.setQueryData(
          ["user", "conversation", variables.conversationId],
          context.previousConversation,
        );
      }
      displayErrorToast(t(I18nKey.CONVERSATION$FAILED_TO_UPDATE_REPOSITORY));
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.CONVERSATION$REPOSITORY_UPDATED));
    },
    onSettled: (data, error, variables) => {
      // Always refetch after error or success
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });
      // Also invalidate the conversations list to update any cached data
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
