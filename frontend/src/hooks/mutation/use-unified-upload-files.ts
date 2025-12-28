import { useMutation } from "@tanstack/react-query";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUploadFiles } from "./use-upload-files";
import { useV1UploadFiles } from "./use-v1-upload-files";
import { FileUploadSuccessResponse } from "#/api/open-hands.types";

interface UnifiedUploadFilesVariables {
  conversationId: string;
  files: File[];
}

/**
 * Unified hook that automatically selects the correct file upload method
 * based on the conversation version (V0 or V1).
 *
 * For V0 conversations: Uses the legacy multi-file upload endpoint
 * For V1 conversations: Uses parallel single-file uploads
 *
 * @returns Mutation hook with the same interface as useUploadFiles
 */
export const useUnifiedUploadFiles = () => {
  const { data: conversation } = useActiveConversation();
  const isV1Conversation = conversation?.conversation_version === "V1";

  // Initialize both hooks
  const v0Upload = useUploadFiles();
  const v1Upload = useV1UploadFiles();

  // Create a unified mutation that delegates to the appropriate hook
  return useMutation({
    mutationKey: ["unified-upload-files"],
    mutationFn: async (
      variables: UnifiedUploadFilesVariables,
    ): Promise<FileUploadSuccessResponse> => {
      const { conversationId, files } = variables;

      if (isV1Conversation) {
        // V1: Use conversation URL and session API key
        return v1Upload.mutateAsync({
          conversationUrl: conversation?.url,
          sessionApiKey: conversation?.session_api_key,
          files,
        });
      }
      // V0: Use conversation ID
      return v0Upload.mutateAsync({
        conversationId,
        files,
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
