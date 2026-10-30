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
  // Tracks whether an IME composition session (Chinese/Japanese/Korean preedit)
  // is currently active. Some third-party Windows IMEs keep the preedit in
  // their own candidate window, so the contenteditable still looks empty —
  // clearing it mid-composition aborts the session and the committed
  // character never lands. We also check this ref in handleKeyDown for the
  // keyCode === 229 case where some IMEs clear the flag before firing the
  // confirmation keystroke.
  const isComposingRef = useRef(false);

  const handleCompositionStart = useCallback(() => {
    isComposingRef.current = true;
  }, []);

  const handleCompositionEnd = useCallback(() => {
    // Reset first; the browser may fire `input` immediately after
    // `compositionend`, and that `input` must NOT skip the clear.
    isComposingRef.current = false;
    // After a non-IME pasted composition (which doesn't fire `input` during
    // the session), manually kick placeholder cleanup once the session ends.
    if (chatInputRef.current) {
      clearEmptyContent(chatInputRef.current, false);
      smartResize();
    }
  }, [chatInputRef, smartResize]);

  // Handle input events
  const handleInput = useCallback(() => {
    smartResize();

    // Clear empty content to ensure placeholder shows — but never during
    // an active IME composition, where clearing would abort the session.
    if (chatInputRef.current) {
      clearEmptyContent(chatInputRef.current, isComposingRef.current);
    }

    // Ensure cursor stays visible when content is scrollable
    ensureCursorVisible(chatInputRef.current);
  }, [smartResize, chatInputRef]);

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

      // Ignore Enter key during IME composition (e.g., Chinese, Japanese,
      // Korean input). When using IME, Enter is used to confirm the
      // composition, not to submit. Two signals matter here:
      //   1. The standard `isComposing` flag (works for Microsoft Pinyin etc.)
      //   2. `keyCode === 229`, which third-party Windows IMEs (Sogou,
      //      WeChat IME, Xinshuru etc.) emit on the confirmation keystroke
      //      even when they've already cleared `isComposing`.
      //   3. Our own `isComposingRef`, which is robust against IMEs that
      //      steal focus to their candidate window (firing blur, which we
      //      used to let the wipe run — now protected by this ref).
      if (
        e.nativeEvent.isComposing ||
        e.nativeEvent.keyCode === 229 ||
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
    // Some third-party Windows IMEs steal focus for their candidate window
    // during composition, firing `blur` while the session is still live.
    // Skip the wipe so it does not abort the preedit.
    if (chatInputRef.current) {
      clearEmptyContent(chatInputRef.current, isComposingRef.current);
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
  };
};
