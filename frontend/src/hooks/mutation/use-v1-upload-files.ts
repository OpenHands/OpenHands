import { useMutation } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { FileUploadSuccessResponse } from "#/api/open-hands.types";

interface V1UploadFilesVariables {
  conversationUrl: string | null | undefined;
  sessionApiKey: string | null | undefined;
  files: File[];
}

/**
 * Hook to upload multiple files in parallel to V1 conversations
 * Uploads files concurrently using Promise.allSettled and aggregates results
 *
 * @returns Mutation hook with mutateAsync function
 */
export const useV1UploadFiles = () =>
  useMutation({
    mutationKey: ["v1-upload-files"],
    mutationFn: async (
      variables: V1UploadFilesVariables,
    ): Promise<FileUploadSuccessResponse> => {
      const { conversationUrl, sessionApiKey, files } = variables;

      // Upload all files in parallel
      const uploadPromises = files.map(async (file) => {
        try {
          // Upload to /workspace/{filename}
          const filePath = `/workspace/${file.name}`;
          await V1ConversationService.uploadFile(
            conversationUrl,
            sessionApiKey,
            file,
            filePath,
          );
          return { success: true as const, fileName: file.name, filePath };
        } catch (error) {
          return {
            success: false as const,
            fileName: file.name,
            filePath: `/workspace/${file.name}`,
            error: error instanceof Error ? error.message : "Unknown error",
          };
        }
      });

      // Wait for all uploads to complete (both successful and failed)
      const results = await Promise.allSettled(uploadPromises);

      // Aggregate the results
      const uploadedFiles: string[] = [];
      const skippedFiles: { name: string; reason: string }[] = [];

      results.forEach((result) => {
        if (result.status === "fulfilled") {
          if (result.value.success) {
            // Return the absolute file path for V1
            uploadedFiles.push(result.value.filePath);
          } else {
            skippedFiles.push({
              name: result.value.fileName,
              reason: result.value.error,
            });
          }
        } else {
          // Promise was rejected (shouldn't happen since we catch errors above)
          skippedFiles.push({
            name: "unknown",
            reason: result.reason?.message || "Upload failed",
          });
        }
      });

      return {
        uploaded_files: uploadedFiles,
        skipped_files: skippedFiles,
      };
    },
    meta: {
      disableToast: true,
    },
  });
