import {
  useMutation,
  useQueryClient,
  type QueryKey,
} from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { I18nKey } from "#/i18n/declaration";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import { Provider } from "#/types/settings";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

interface UpdateRepositoryVariables {
  conversationId: string;
  repository: string | null;
  branch?: string | null;
  gitProvider?: Provider | null;
}

export const useUpdateConversationRepository = () => {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    mutationFn: (variables: UpdateRepositoryVariables) =>
      AgentServerConversationService.updateConversationRepository(
        variables.conversationId,
        variables.repository,
        variables.branch,
        variables.gitProvider,
      ),
    onMutate: async (variables) => {
      // The active conversation is cached under a prefix-extended key
      // (`["user", "conversation", id, backendId, orgId]`) by
      // `useUserConversation`. Cancel/snapshot/update via the prefix so
      // the optimistic change actually reaches the rendered query, not a
      // stale 3-element key that no observer ever reads.
      const prefix: QueryKey = [
        "user",
        "conversation",
        variables.conversationId,
      ];
      await queryClient.cancelQueries({ queryKey: prefix });

      const previousEntries =
        queryClient.getQueriesData<AppConversation | null>({
          queryKey: prefix,
        });

      queryClient.setQueriesData<AppConversation | null>(
        { queryKey: prefix },
        (old) =>
          old
            ? {
                ...old,
                selected_repository: variables.repository,
                selected_branch: variables.branch ?? null,
                git_provider: variables.gitProvider ?? null,
              }
            : old,
      );

      return { previousEntries };
    },
    onError: (err, variables, context) => {
      context?.previousEntries.forEach(([key, data]) => {
        queryClient.setQueryData(key, data);
      });
      displayErrorToast(t(I18nKey.CONVERSATION$FAILED_TO_UPDATE_REPOSITORY));
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.CONVERSATION$REPOSITORY_UPDATED));
    },
    onSettled: (data, error, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", variables.conversationId],
      });
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
      // The local-git-info probe is keyed off the conversation's
      // working_dir; force a re-probe so the connected-repo row
      // re-renders without waiting on the next 10s poll tick.
      queryClient.invalidateQueries({
        queryKey: ["local-git-info", variables.conversationId],
      });
    },
  });
};
