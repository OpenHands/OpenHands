import { useMutation } from "@tanstack/react-query";
import { useTracking } from "#/hooks/use-tracking";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { downloadBlob } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

export const useDownloadConversation = () => {
  const { trackDownloadTrajectoryButtonClicked } = useTracking();
  const { t } = useTranslation("openhands");

  return useMutation({
    mutationKey: ["conversations", "download"],
    mutationFn: async (conversationId: string) => {
      trackDownloadTrajectoryButtonClicked();
      const blob =
        await AgentServerConversationService.downloadConversation(
          conversationId,
        );
      downloadBlob(blob, `conversation_${conversationId}.zip`);
    },
    onError: () => {
      displayErrorToast(t(I18nKey.CONVERSATION$DOWNLOAD_ERROR));
    },
  });
};
