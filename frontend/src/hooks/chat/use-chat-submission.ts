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
  onSubmit: (message: string) => Promise<void> | void,
  resetManualResize?: () => void,
) => {
  // Send button click handler
  const handleSubmit = useCallback(async () => {
    const message = chatInputRef.current?.innerText || "";
    const trimmedMessage = message.trim();

    if (!trimmedMessage) {
      return;
    }

    try {
      await onSubmit(message);

      // Clear the input only after a successful send so the user's text is
      // preserved if the network call fails (e.g. "Failed to connect to server").
      clearTextContent(chatInputRef.current);
      clearFileInput(fileInputRef.current);

      // Reset height and show suggestions again
      smartResize();

      // Reset manual resize state for next message
      resetManualResize?.();
    } catch {
      // Send failed — leave the input intact so the user doesn't lose their message.
    }
  }, [chatInputRef, fileInputRef, smartResize, onSubmit, resetManualResize]);

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
