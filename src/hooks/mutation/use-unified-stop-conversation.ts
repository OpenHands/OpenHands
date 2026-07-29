import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { useTranslation } from "react-i18next";
import {
  TOAST_OPTIONS,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
import { useNavigation } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import { ExecutionStatus } from "#/types/agent-server/core";
import {
  pauseConversation,
  patchConversationInCache,
} from "./conversation-mutation-utils";

export const useUnifiedPauseConversation = () => {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { conversationId: currentConversationId, navigate } = useNavigation();

  return useMutation({
    mutationKey: ["stop-conversation"],
    mutationFn: async (variables: { conversationId: string }) =>
      pauseConversation(variables.conversationId),
    onMutate: async () => {
      const toastId = toast.loading(
        t(I18nKey.TOAST$STOPPING_CONVERSATION),
        TOAST_OPTIONS,
      );

      await queryClient.cancelQueries({ queryKey: ["user", "conversations"] });
      const previousConversations = queryClient.getQueryData([
        "user",
        "conversations",
      ]);

      return { previousConversations, toastId };
    },
    onError: (_, __, context) => {
      if (context?.toastId) {
        toast.dismiss(context.toastId);
      }
      displayErrorToast(t(I18nKey.TOAST$FAILED_TO_STOP_CONVERSATION));

      if (context?.previousConversations) {
        queryClient.setQueryData(
          ["user", "conversations"],
          context.previousConversations,
        );
      }
    },
    onSuccess: (_, variables, context) => {
      if (context?.toastId) {
        toast.dismiss(context.toastId);
      }
      toast.success(t(I18nKey.TOAST$CONVERSATION_STOPPED), TOAST_OPTIONS);

      // Update both execution_status and sandbox_status together so that
      // WebSocketProviderWrapper's sandbox_status === "PAUSED" gate fires
      // immediately when the user reopens this conversation — preventing a
      // WebSocket connection attempt against the now-paused sandbox host
      // before the next useActiveConversation poll returns.
      patchConversationInCache(queryClient, variables.conversationId, {
        execution_status: ExecutionStatus.PAUSED,
        sandbox_status: "PAUSED",
      });

      if (currentConversationId === variables.conversationId) {
        navigate("/conversations");
      }
    },
  });
};
