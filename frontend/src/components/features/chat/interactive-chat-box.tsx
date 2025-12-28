import { isFileImage } from "#/utils/is-file-image";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { validateFiles } from "#/utils/file-validation";
import { CustomChatInput } from "./custom-chat-input";
import { AgentState } from "#/types/agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { GitControlBar } from "./git-control-bar";
import { useConversationStore } from "#/stores/conversation-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { processFiles, processImages } from "#/utils/file-processing";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import { isTaskPolling } from "#/utils/utils";

interface InteractiveChatBoxProps {
  onSubmit: (message: string, images: File[], files: File[]) => void;
}

export function InteractiveChatBox({ onSubmit }: InteractiveChatBoxProps) {
  const {
    images,
    files,
    addImages,
    addFiles,
    clearAllFiles,
    addFileLoading,
    removeFileLoading,
    addImageLoading,
    removeImageLoading,
    subConversationTaskId,
  } = useConversationStore();
  const { curAgentState } = useAgentState();
  const { data: conversation } = useActiveConversation();

  // Poll sub-conversation task to check if it's loading
  const { taskStatus: subConversationTaskStatus } =
    useSubConversationTaskPolling(
      subConversationTaskId,
      conversation?.conversation_id || null,
    );

  // Helper function to validate and filter files
  const validateAndFilterFiles = (selectedFiles: File[]) => {
    const validation = validateFiles(selectedFiles, [...images, ...files]);

    if (!validation.isValid) {
      displayErrorToast(`Error: ${validation.errorMessage}`);
      return null;
    }

    const validFiles = selectedFiles.filter((f) => !isFileImage(f));
    const validImages = selectedFiles.filter((f) => isFileImage(f));

    return { validFiles, validImages };
  };

  // Helper function to show loading indicators for files
  const showLoadingIndicators = (validFiles: File[], validImages: File[]) => {
    validFiles.forEach((file) => addFileLoading(file.name));
    validImages.forEach((image) => addImageLoading(image.name));
  };

  // Helper function to handle successful file processing results
  const handleSuccessfulFiles = (fileResults: { successful: File[] }) => {
    if (fileResults.successful.length > 0) {
      addFiles(fileResults.successful);
      fileResults.successful.forEach((file) => removeFileLoading(file.name));
    }
  };

  // Helper function to handle successful image processing results
  const handleSuccessfulImages = (imageResults: { successful: File[] }) => {
    if (imageResults.successful.length > 0) {
      addImages(imageResults.successful);
      imageResults.successful.forEach((image) =>
        removeImageLoading(image.name),
      );
    }
  };

  // Helper function to handle failed file processing results
  const handleFailedFiles = (
    fileResults: { failed: { file: File; error: Error }[] },
    imageResults: { failed: { file: File; error: Error }[] },
  ) => {
    fileResults.failed.forEach(({ file, error }) => {
      removeFileLoading(file.name);
      displayErrorToast(
        `Failed to process file ${file.name}: ${error.message}`,
      );
    });

    imageResults.failed.forEach(({ file, error }) => {
      removeImageLoading(file.name);
      displayErrorToast(
        `Failed to process image ${file.name}: ${error.message}`,
      );
    });
  };

  // Helper function to clear loading states on error
  const clearLoadingStates = (validFiles: File[], validImages: File[]) => {
    validFiles.forEach((file) => removeFileLoading(file.name));
    validImages.forEach((image) => removeImageLoading(image.name));
  };

  const handleUpload = async (selectedFiles: File[]) => {
    // Step 1: Validate and filter files
    const result = validateAndFilterFiles(selectedFiles);
    if (!result) return;

    const { validFiles, validImages } = result;

    // Step 2: Show loading indicators immediately
    showLoadingIndicators(validFiles, validImages);

    // Step 3: Process files using REAL FileReader
    try {
      const [fileResults, imageResults] = await Promise.all([
        processFiles(validFiles),
        processImages(validImages),
      ]);

      // Step 4: Handle successful results
      handleSuccessfulFiles(fileResults);
      handleSuccessfulImages(imageResults);

      // Step 5: Handle failed results
      handleFailedFiles(fileResults, imageResults);
    } catch {
      // Clear loading states and show error
      clearLoadingStates(validFiles, validImages);
      displayErrorToast("An unexpected error occurred while processing files");
    }
  };

  const handleSubmit = (message: string) => {
    onSubmit(message, images, files);
    clearAllFiles();
  };

  const handleSuggestionsClick = (suggestion: string) => {
    handleSubmit(suggestion);
  };

  const isDisabled =
    curAgentState === AgentState.LOADING ||
    curAgentState === AgentState.AWAITING_USER_CONFIRMATION ||
    isTaskPolling(subConversationTaskStatus);

  return (
    <div data-testid="interactive-chat-box">
      <CustomChatInput
        disabled={isDisabled}
        onSubmit={handleSubmit}
        onFilesPaste={handleUpload}
        conversationStatus={conversation?.status || null}
      />
      <div className="mt-4">
        <GitControlBar onSuggestionsClick={handleSuggestionsClick} />
      </div>
    </div>
  );
}
