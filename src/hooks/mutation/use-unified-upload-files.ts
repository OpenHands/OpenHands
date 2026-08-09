import { useMutation, useQueryClient } from "@tanstack/react-query";
import { uploadFilesToConversation } from "#/api/conversation-file-upload.api";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { WORKSPACE_QUERY_KEYS } from "#/hooks/query/query-keys";
import { FileUploadSuccessResponse } from "#/api/open-hands.types";

interface UnifiedUploadFilesVariables {
  conversationId: string;
  files: File[];
}

/**
 * Uploads files for the active conversation (local agent-server or cloud runtime).
 */
export const useUnifiedUploadFiles = () => {
  const { data: conversation } = useActiveConversation();
  const queryClient = useQueryClient();

  return useMutation({
    mutationKey: ["unified-upload-files"],
    mutationFn: async (
      variables: UnifiedUploadFilesVariables,
    ): Promise<FileUploadSuccessResponse> => {
      const { conversationId, files } = variables;
      return uploadFilesToConversation(conversationId, files, conversation);
    },
    onSuccess: (data, variables) => {
      if (data.uploaded_files.length > 0) {
        const { conversationId } = variables;
        queryClient.invalidateQueries({
          queryKey:
            WORKSPACE_QUERY_KEYS.fileChangesForConversation(conversationId),
        });
        queryClient.invalidateQueries({
          queryKey: WORKSPACE_QUERY_KEYS.filesForConversation(conversationId),
        });
      }
    },
    meta: {
      disableToast: true,
    },
  });
};
