import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { I18nKey } from "#/i18n/declaration";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";

export const useUpdateConversationPublicFlag = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: (variables: { conversationId: string; isPublic: boolean }) =>
      V1ConversationService.updateConversationPublicFlag(
        variables.conversationId,
        variables.isPublic,
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
            ? { ...old, public: variables.isPublic }
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
      displayErrorToast(
        t(I18nKey.CONVERSATION$FAILED_TO_UPDATE_PUBLIC_SHARING),
      );
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.CONVERSATION$PUBLIC_SHARING_UPDATED));
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
