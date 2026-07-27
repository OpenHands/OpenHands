import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import toast from "react-hot-toast";
import { I18nKey } from "#/i18n/declaration";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  displayErrorToast,
  displaySuccessToast,
  TOAST_OPTIONS,
} from "#/utils/custom-toast-handlers";
import { useNavigation } from "#/context/navigation-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useTracking } from "#/hooks/use-tracking";

export const useNewConversationCommand = () => {
  const queryClient = useQueryClient();
  const { navigate } = useNavigation();
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { trackConversationCreated } = useTracking();

  const mutation = useMutation({
    mutationFn: async () => {
      if (!conversation?.id) {
        throw new Error("No active conversation");
      }

      // /new reuses the parent conversation's sandbox (matches OpenHands
      // cloud behavior); it is NOT a sub-conversation, so parent_conversation_id
      // and agent_type stay undefined.
      const startTask = await AgentServerConversationService.createConversation(
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        conversation.sandbox_id ?? undefined,
      );

      if (startTask.status === "ERROR") {
        throw new Error(
          startTask.detail || "Failed to create new conversation",
        );
      }

      // Cloud returns a WORKING task (no app_conversation_id yet);
      // navigate to /conversations/task-{id} so useTaskPolling drives it
      // to READY. Local creates synchronously — app_conversation_id is
      // already set, so we navigate straight to it.
      const newConversationId = startTask.app_conversation_id
        ? startTask.app_conversation_id
        : `task-${startTask.id}`;

      return {
        newConversationId,
        oldConversationId: conversation.id,
        taskId: startTask.id,
      };
    },
    onMutate: () => {
      toast.loading(t(I18nKey.CONVERSATION$CLEARING), {
        ...TOAST_OPTIONS,
        id: "clear-conversation",
      });
    },
    onSuccess: (data) => {
      trackConversationCreated({
        conversationId: data.newConversationId,
        taskId: data.taskId,
        hasRepository: false,
        hasWorkspace: false,
        hasInitialQuery: false,
        hasParentConversation: false,
        entryPoint: "new_command",
      });

      toast.dismiss("clear-conversation");
      displaySuccessToast(t(I18nKey.CONVERSATION$CLEAR_SUCCESS));
      navigate(`/conversations/${data.newConversationId}`);

      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
      queryClient.invalidateQueries({
        queryKey: ["v1-batch-get-app-conversations"],
      });
    },
    onError: (error) => {
      toast.dismiss("clear-conversation");
      let clearError = t(I18nKey.CONVERSATION$CLEAR_UNKNOWN_ERROR);
      if (error instanceof Error) {
        clearError = error.message;
      } else if (typeof error === "string") {
        clearError = error;
      }
      displayErrorToast(
        t(I18nKey.CONVERSATION$CLEAR_FAILED, { error: clearError }),
      );
    },
  });

  return mutation;
};
