import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import ProjectsService from "#/api/projects-service/projects-service.api";
import { useChatAttachmentUpload } from "#/hooks/chat/use-chat-attachment-upload";
import { I18nKey } from "#/i18n/declaration";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Adds one of the project's files to the composer like a file from disk: an
 * image is embedded in the message, any other file is uploaded to the
 * conversation's workspace when the message is sent. Takes the file's name.
 */
export function useAttachProjectFile(
  projectId: string,
  { onSuccess }: { onSuccess?: () => void } = {},
) {
  const { t } = useTranslation("openhands");
  const { handleUpload } = useChatAttachmentUpload();
  return useMutation({
    mutationFn: async (name: string) => {
      const blob = await ProjectsService.downloadArtifact(projectId, name);
      await handleUpload([new File([blob], name, { type: blob.type })]);
    },
    meta: { disableToast: true },
    onSuccess,
    onError: (error) =>
      displayApiErrorToast(error, t(I18nKey.PROJECTS$ATTACH_FAILED)),
  });
}
