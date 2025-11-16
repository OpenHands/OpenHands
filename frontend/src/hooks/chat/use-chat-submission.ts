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

    // Remove @ symbols ONLY from file paths inserted by autocomplete
    // Only strips @ if the token:
    //   1. Contains a forward slash (e.g., @src/file.ts, @path/to/file)
    //   2. Starts with ./, ../, or ~/ (e.g., @./file.ts, @../parent/file)
    //   3. Has a common file extension (e.g., @file.py, @component.tsx)
    // This preserves @ in code like @property, @dataclass, @Override, etc.
    const cleanedMessage = message.replace(
      /(^|\s)@((?:\.\/|\.\.\/|~\/)[^\s]*|[^\s]*\/[^\s]*|[^\s]+\.(?:ts|tsx|js|jsx|py|java|cpp|c|h|hpp|cs|rb|go|rs|md|txt|json|yaml|yml|xml|html|css|scss|sass|less|vue|svelte)(?:\s|$))/gi,
      "$1$2",
    );

    onSubmit(cleanedMessage);

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

    // Remove @ symbols ONLY from file paths inserted by autocomplete
    // Only strips @ if the token:
    //   1. Contains a forward slash (e.g., @src/file.ts, @path/to/file)
    //   2. Starts with ./, ../, or ~/ (e.g., @./file.ts, @../parent/file)
    //   3. Has a common file extension (e.g., @file.py, @component.tsx)
    // This preserves @ in code like @property, @dataclass, @Override, etc.
    const cleanedMessage = message.replace(
      /(^|\s)@((?:\.\/|\.\.\/|~\/)[^\s]*|[^\s]*\/[^\s]*|[^\s]+\.(?:ts|tsx|js|jsx|py|java|cpp|c|h|hpp|cs|rb|go|rs|md|txt|json|yaml|yml|xml|html|css|scss|sass|less|vue|svelte)(?:\s|$))/gi,
      "$1$2",
    );

    onSubmit(cleanedMessage.trim());

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
