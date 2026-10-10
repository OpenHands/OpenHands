import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { downloadConversationFile } from "#/api/conversation-file-download.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { downloadBlob } from "#/utils/utils";

// @spec FD-001 — Keep the download tied to the file selected at click time
export function useDownloadWorkspaceFile() {
  const { t } = useTranslation("openhands");
  return useMutation({
    mutationFn: async ({
      conversation,
      path,
    }: {
      conversation: AppConversation;
      path: string;
    }) => {
      const blob = await downloadConversationFile(conversation, path);
      downloadBlob(blob, path.split("/").pop() || path);
    },
    onError: () => displayErrorToast(t(I18nKey.FILES$DOWNLOAD_ERROR)),
  });
}
