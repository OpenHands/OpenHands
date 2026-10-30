import { useCallback, useRef } from "react";
import { isMobileUserAgent } from "#/utils/utils";
import {
  ensureCursorVisible,
  clearEmptyContent,
  getClipboardFiles,
} from "#/components/features/chat/utils/chat-input.utils";

/**
 * Hook for handling chat input events
 */
export const useChatInputEvents = (
  chatInputRef: React.RefObject<HTMLDivElement | null>,
  smartResize: () => void,
  increaseHeightForEmptyContent: () => void,
  checkIsContentEmpty: () => boolean,
  onFocus?: () => void,
  onBlur?: () => void,
) => {
  const isComposingRef = useRef(false);

  const handleCompositionStart = useCallback(() => {
    isComposingRef.current = true;
    if (chatInputRef.current) {
      chatInputRef.current.setAttribute("data-composing", "true");
    }
  }, [chatInputRef]);

  const handleCompositionEnd = useCallback(() => {
    isComposingRef.current = false;
    if (chatInputRef.current) {
      chatInputRef.current.removeAttribute("data-composing");
      clearEmptyContent(chatInputRef.current);
    }
    smartResize();
  }, [chatInputRef, smartResize]);

  // Handle input events
  const handleInput = useCallback(
    (e?: React.FormEvent<HTMLDivElement>) => {
      smartResize();

      const nativeEvent = e?.nativeEvent as InputEvent | undefined;
      const isComposing =
        isComposingRef.current || Boolean(nativeEvent?.isComposing);

      // Clear empty content to ensure placeholder shows, but never wipe
      // while an IME composition session is active because assigning
      // innerHTML/textContent aborts composition for third-party Windows IMEs.
      if (chatInputRef.current && !isComposing) {
        clearEmptyContent(chatInputRef.current);
      }

      // Ensure cursor stays visible when content is scrollable
      ensureCursorVisible(chatInputRef.current);
    },
    [smartResize, chatInputRef],
  );

  // Handle paste events to clean up formatting and handle files
  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      e.preventDefault();

      const files = getClipboardFiles(e.clipboardData);
      const hasFiles = files.length > 0;

      if (hasFiles) {
        // Handle file paste - let the file handling system process the files
        // We'll trigger a custom event that the file handling system can listen to
        const customEvent = new CustomEvent("pasteFiles", {
          detail: { files },
        });
        document.dispatchEvent(customEvent);
        return;
      }

      // Handle text paste as before
      const text = e.clipboardData.getData("text/plain");
      if (text) {
        // Insert plain text
        document.execCommand("insertText", false, text);
        // Trigger resize
        setTimeout(smartResize, 0);
      }
    },
    [smartResize],
  );

  // Handle key events
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, disabled: boolean, handleSubmit: () => void) => {
      if (e.key !== "Enter") {
        return;
      }

      // Ignore Enter key during IME composition (e.g., Chinese, Japanese, Korean input).
      // When using IME, Enter is used to confirm the composition, not to submit.
      // On Windows with third-party IMEs, e.nativeEvent.isComposing might be false
      // on the confirmation keystroke while keyCode is 229 or composition is active.
      if (
        e.nativeEvent.isComposing ||
        e.keyCode === 229 ||
        isComposingRef.current
      ) {
        return;
      }

      if (checkIsContentEmpty()) {
        e.preventDefault();
        increaseHeightForEmptyContent();
        return;
      }

      // Submit on Enter for everything except phones/tablets (where Enter
      // inserts a newline and the user taps the send button). We gate on the
      // user agent rather than touch capability so a desktop OS with a
      // touchscreen (e.g. a Windows 2-in-1) still submits on Enter.
      if (!isMobileUserAgent() && !e.shiftKey && !disabled) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [checkIsContentEmpty, increaseHeightForEmptyContent],
  );

  // Handle blur events to ensure placeholder shows when empty
  const handleBlur = useCallback(() => {
    // Clear empty content to ensure placeholder shows, but skip wiping
    // if an IME composition session is still active (e.g. third-party candidate popup).
    if (chatInputRef.current && !isComposingRef.current) {
      clearEmptyContent(chatInputRef.current);
    }

    // Call the original onBlur callback if provided
    if (onBlur) {
      onBlur();
    }
  }, [chatInputRef, onBlur]);

  // Handle focus events
  const handleFocus = useCallback(() => {
    if (onFocus) {
      onFocus();
    }
  }, [onFocus]);

  return {
    handleInput,
    handlePaste,
    handleKeyDown,
    handleBlur,
    handleFocus,
    handleCompositionStart,
    handleCompositionEnd,
    isComposingRef,
  };
};
