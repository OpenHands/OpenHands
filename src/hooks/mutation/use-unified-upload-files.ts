import { useMutation } from "@tanstack/react-query";
import {
  uploadFilesToConversation,
  type UploadProgressInfo,
} from "#/api/conversation-file-upload.api";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { FileUploadSuccessResponse } from "#/api/open-hands.types";

interface UnifiedUploadFilesVariables {
  conversationId: string;
  files: File[];
  /** Optional callback invoked after each upload batch with current progress. */
  onProgress?: (info: UploadProgressInfo) => void;
}

/**
 * Uploads files for the active conversation (local agent-server or cloud runtime).
 */
export const useUnifiedUploadFiles = () => {
  const { data: conversation } = useActiveConversation();

  return useMutation({
    mutationKey: ["unified-upload-files"],
    mutationFn: async (
      variables: UnifiedUploadFilesVariables,
    ): Promise<FileUploadSuccessResponse> => {
      const { conversationId, files, onProgress } = variables;
      return uploadFilesToConversation(
        conversationId,
        files,
        conversation,
        onProgress,
      );
    },
    meta: {
      disableToast: true,
    },
  });
};
