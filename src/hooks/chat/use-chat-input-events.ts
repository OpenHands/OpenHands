import { useCallback, useRef } from "react";
import { isMobileUserAgent } from "#/utils/utils";
import {
  ensureCursorVisible,
  clearEmptyContent,
  clearImeComposing,
  getClipboardFiles,
  isImeComposingEvent,
  markImeComposing,
  type ImeEventInput,
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
  // Some Windows IMEs omit `isComposing` on individual events; the session
  // flag covers compositionstart → compositionend.
  const composingRef = useRef(false);

  const isImeSessionActive = useCallback(
    (event?: ImeEventInput) => isImeComposingEvent(event, composingRef.current),
    [],
  );

  const handleCompositionStart = useCallback(() => {
    composingRef.current = true;
    markImeComposing(chatInputRef.current);
  }, [chatInputRef]);

  const handleCompositionEnd = useCallback(() => {
    composingRef.current = false;
    clearImeComposing(chatInputRef.current);
    smartResize();
  }, [chatInputRef, smartResize]);

  // Handle input events. Mutating the editor during IME composition (even to
  // "clear empty" markup) cancels the composition, which is why third-party
  // Chinese IMEs can appear to type nothing while the OS IME still works.
  const handleInput = useCallback(
    (e?: React.FormEvent<HTMLDivElement>) => {
      const composing = isImeComposingEvent(e, composingRef.current);
      smartResize();

      if (composing) {
        return;
      }

      if (chatInputRef.current) {
        clearEmptyContent(chatInputRef.current);
      }

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

      // Ignore Enter during IME composition (confirm candidate, not submit).
      // Also treat keyCode 229: some Windows IMEs leave isComposing false.
      if (isImeComposingEvent(e, composingRef.current)) {
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

  // Handle blur events to ensure placeholder shows when empty.
  // Third-party IMEs may steal focus for their candidate window; delay the
  // DOM cleanup so a composing session is not aborted on that spurious blur.
  const handleBlur = useCallback(() => {
    if (onBlur) {
      onBlur();
    }

    window.setTimeout(() => {
      if (composingRef.current) {
        return;
      }
      if (
        chatInputRef.current &&
        document.activeElement === chatInputRef.current
      ) {
        return;
      }
      clearEmptyContent(chatInputRef.current);
    }, 0);
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
    isImeSessionActive,
  };
};
