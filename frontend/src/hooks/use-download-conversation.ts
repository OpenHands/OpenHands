import { useMutation } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import { useTranslation } from "react-i18next";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { downloadBlob } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

export const useDownloadConversation = () => {
  const posthog = usePostHog();
  const { t } = useTranslation();

  return useMutation({
    mutationKey: ["conversations", "download"],
    mutationFn: async (conversationId: string) => {
      posthog.capture("download_trajectory_button_clicked");
      const blob =
        await V1ConversationService.downloadConversation(conversationId);
      downloadBlob(blob, `conversation_${conversationId}.zip`);
    },
    onError: () => {
      displayErrorToast(t(I18nKey.CONVERSATION$DOWNLOAD_ERROR));
    },
  });
};
