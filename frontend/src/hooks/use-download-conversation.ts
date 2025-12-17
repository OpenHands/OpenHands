import { usePostHog } from "posthog-js/react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export const useDownloadConversation = () => {
  const [pending, setPending] = useState(false);
  const posthog = usePostHog();
  const { t } = useTranslation();

  return {
    pending,
    downloadConversation: async (conversationId: string) => {
      try {
        setPending(true);
        posthog.capture("download_trajectory_button_clicked");
        const blob =
          await V1ConversationService.downloadConversation(conversationId);
        // Create a download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `conversation_${conversationId}.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } catch (error) {
        displayErrorToast(t(I18nKey.CONVERSATION$DOWNLOAD_ERROR));
      } finally {
        setPending(false);
      }
    },
  };
};
