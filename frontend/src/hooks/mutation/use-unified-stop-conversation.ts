import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router";
import toast from "react-hot-toast";
import { useTranslation } from "react-i18next";
import { TOAST_OPTIONS } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import {
  pauseV1ConversationSandbox,
  updateConversationSandboxStatusInCache,
} from "./conversation-mutation-utils";

/**
 * Hook to pause a conversation sandbox.
 *
 * Usage:
 * const { mutate: stopConversation } = useUnifiedPauseConversationSandbox();
 * stopConversation({ conversationId: "some-id" });
 */
export const useUnifiedPauseConversationSandbox = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const params = useParams<{ conversationId: string }>();

  return useMutation({
    mutationKey: ["stop-conversation"],
    mutationFn: async (variables: { conversationId: string }) =>
      pauseV1ConversationSandbox(variables.conversationId),
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
      toast.error(t(I18nKey.TOAST$FAILED_TO_STOP_CONVERSATION), TOAST_OPTIONS);

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

      updateConversationSandboxStatusInCache(
        queryClient,
        variables.conversationId,
        "PAUSED",
      );

      // Only redirect if we're stopping the conversation we're currently viewing
      if (params.conversationId === variables.conversationId) {
        navigate("/");
      }
    },
  });
};
