import { useCallback } from "react";
import {
  clearTextContent,
  clearFileInput,
} from "#/components/features/chat/utils/chat-input.utils";

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
  // Send button click handler
  const handleSubmit = useCallback(() => {
    const message = chatInputRef.current?.innerText || "";
    const trimmedMessage = message.trim();

    if (!trimmedMessage) {
      return;
    }

    onSubmit(message);

    // Clear the input
    clearTextContent(chatInputRef.current);
    clearFileInput(fileInputRef.current);

    // Reset height and show suggestions again
    smartResize();

    // Reset manual resize state for next message
    resetManualResize?.();
  }, [chatInputRef, fileInputRef, smartResize, onSubmit, resetManualResize]);

  // Resume agent button click handler
  const handleResumeAgent = useCallback(() => {
    const message = chatInputRef.current?.innerText || "continue";

    onSubmit(message.trim());

    // Clear the input
    clearTextContent(chatInputRef.current);
    clearFileInput(fileInputRef.current);

    // Reset height and show suggestions again
    smartResize();

    // Reset manual resize state for next message
    resetManualResize?.();
  }, [chatInputRef, fileInputRef, smartResize, onSubmit, resetManualResize]);

  // Handle stop button click
  const handleStop = useCallback((onStop?: () => void) => {
    if (onStop) {
      onStop();
    }
  }, []);

  return {
    handleSubmit,
    handleResumeAgent,
    handleStop,
  };
};
