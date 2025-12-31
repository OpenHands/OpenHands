import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router";
import toast from "react-hot-toast";
import { useTranslation } from "react-i18next";
import { TOAST_OPTIONS } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import {
  getConversationVersionFromQueryCache,
  pauseV1ConversationSandbox,
  stopV0Conversation,
  updateConversationStatusInCache,
  invalidateConversationQueries,
} from "./conversation-mutation-utils";

/**
 * Unified hook that automatically routes to the correct pause conversation sandbox
 * implementation based on the conversation version (V0 or V1).
 *
 * This hook checks the cached conversation data to determine the version, then calls
 * the appropriate API directly. Returns a single useMutation instance that all components share.
 *
 * Usage is the same as useStopConversation:
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
    mutationFn: async (variables: {
      conversationId: string;
      version?: "V0" | "V1";
    }) => {
      // Use provided version or fallback to cache lookup
      const version =
        variables.version ||
        getConversationVersionFromQueryCache(
          queryClient,
          variables.conversationId,
        );

      if (version === "V1") {
        return pauseV1ConversationSandbox(variables.conversationId);
      }

      return stopV0Conversation(variables.conversationId);
    },
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
    onSettled: (_, __, variables) => {
      invalidateConversationQueries(queryClient, variables.conversationId);
    },
    onSuccess: (_, variables, context) => {
      if (context?.toastId) {
        toast.dismiss(context.toastId);
      }
      toast.success(t(I18nKey.TOAST$CONVERSATION_STOPPED), TOAST_OPTIONS);

      updateConversationStatusInCache(
        queryClient,
        variables.conversationId,
        "STOPPED",
      );

      // Only redirect if we're stopping the conversation we're currently viewing
      if (params.conversationId === variables.conversationId) {
        navigate("/");
      }
    },
  });
};
