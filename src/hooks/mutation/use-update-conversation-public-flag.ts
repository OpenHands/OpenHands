import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export const useUpdateConversationPublicFlag = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: (variables: { conversationId: string; isPublic: boolean }) =>
      AgentServerConversationService.updateConversationPublicFlag(
        variables.conversationId,
        variables.isPublic,
      ),
    onMutate: async (variables) => {
      const conversationQueryKey = [
        "user",
        "conversation",
        variables.conversationId,
      ] as const;

      await queryClient.cancelQueries({ queryKey: conversationQueryKey });

      const previousEntries = queryClient.getQueriesData({
        queryKey: conversationQueryKey,
      });

      queryClient.setQueriesData(
        { queryKey: conversationQueryKey },
        (old: unknown) =>
          old && typeof old === "object"
            ? { ...old, public: variables.isPublic }
            : old,
      );

      return { previousEntries };
    },
    onError: (_err, variables, context) => {
      context?.previousEntries?.forEach(([queryKey, data]) => {
        queryClient.setQueryData(queryKey, data);
      });
      displayErrorToast(
        t(I18nKey.CONVERSATION$FAILED_TO_UPDATE_PUBLIC_SHARING),
      );
    },
    onSettled: (_data, _error, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
    },
  });
};
