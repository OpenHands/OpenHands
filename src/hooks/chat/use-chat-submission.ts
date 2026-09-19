import { useCallback, useEffect, useState } from "react";
import {
  clearTextContent,
  clearFileInput,
} from "#/components/features/chat/utils/chat-input.utils";
import { useConversationStore } from "#/stores/conversation-store";

/**
 * Hook for handling chat message submission
 */
export const useChatSubmission = (
  chatInputRef: React.RefObject<HTMLDivElement | null>,
  fileInputRef: React.RefObject<HTMLInputElement | null>,
  smartResize: () => void,
  onSubmit: (message: string) => void,
  resetManualResize?: () => void,
) => {
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const loadingFiles = useConversationStore((state) => state.loadingFiles);
  const loadingImages = useConversationStore((state) => state.loadingImages);

  const isUploading = loadingFiles.length > 0 || loadingImages.length > 0;

  const completeSubmit = useCallback(
    (message: string) => {
      onSubmit(message);

      // Clear the input
      clearTextContent(chatInputRef.current);
      clearFileInput(fileInputRef.current);

      // Reset height and show suggestions again
      smartResize();

      // Reset manual resize state for next message
      resetManualResize?.();
    },
    [chatInputRef, fileInputRef, smartResize, onSubmit, resetManualResize],
  );

  // Send a message that was requested while files were still uploading.
  useEffect(() => {
    if (pendingMessage === null || isUploading) {
      return;
    }

    const message = pendingMessage;
    setPendingMessage(null);
    completeSubmit(message);
  }, [pendingMessage, isUploading, completeSubmit]);

  // Send button click handler
  const handleSubmit = useCallback(() => {
    const message = chatInputRef.current?.innerText || "";
    const trimmedMessage = message.trim();
    const { images, files } = useConversationStore.getState();
    const hasAttachments = images.length > 0 || files.length > 0;

    if (!trimmedMessage && !hasAttachments) {
      return;
    }

    // Keep the prompt visible until all attachments finish processing.
    if (isUploading) {
      setPendingMessage(message);
      return;
    }

    completeSubmit(message);
  }, [chatInputRef, isUploading, completeSubmit]);

  // Handle stop button click
  const handleStop = useCallback((onStop?: () => void) => {
    if (onStop) {
      onStop();
    }
  }, []);

  return {
    handleSubmit,
    handleStop,
  };
};
