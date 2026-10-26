import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  TOAST_OPTIONS,
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import type { WakeableConversation } from "#/utils/wakeable-conversations";
import {
  invalidateConversationQueries,
  wakeConversation,
} from "./conversation-mutation-utils";

/**
 * Resume every sleeping conversation in one action ("wake all agents"):
 * each one independently, so a single unreachable sandbox does not block the
 * rest. Failures are reported as a count instead of one toast per sandbox.
 */
export const useWakeAllConversations = () => {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();

  return useMutation({
    mutationKey: ["wake-all-conversations"],
    mutationFn: async (variables: {
      conversations: WakeableConversation[];
    }) => {
      const results = await Promise.allSettled(
        variables.conversations.map((conversation) =>
          wakeConversation(conversation),
        ),
      );
      return {
        failedCount: results.filter((result) => result.status === "rejected")
          .length,
      };
    },
    onMutate: (variables) => {
      const toastId = toast.loading(
        t(I18nKey.TOAST$WAKING_ALL_AGENTS, {
          count: variables.conversations.length,
        }),
        TOAST_OPTIONS,
      );
      return { toastId };
    },
    onSuccess: ({ failedCount }, variables, context) => {
      if (context?.toastId) {
        toast.dismiss(context.toastId);
      }
      if (failedCount === 0) {
        displaySuccessToast(t(I18nKey.TOAST$ALL_AGENTS_WOKEN));
      } else {
        displayErrorToast(
          t(I18nKey.TOAST$FAILED_TO_WAKE_ALL_AGENTS, { count: failedCount }),
        );
      }
      for (const conversation of variables.conversations) {
        invalidateConversationQueries(queryClient, conversation.id);
      }
    },
  });
};
