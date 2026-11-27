import type { RefObject } from "react";
import { useState, useCallback, useEffect } from "react";
import { useAutoResize } from "#/hooks/use-auto-resize";
import { isBottomAnchored } from "#/hooks/use-drag-resize";
import { CHAT_INPUT } from "#/utils/constants";
import {
  IMessageToSend,
  useConversationStore,
} from "#/stores/conversation-store";

/**
 * Hook for managing grip resize functionality
 */
export const useGripResize = (
  chatInputRef: RefObject<HTMLDivElement | null>,
  messageToSend: IMessageToSend | null,
) => {
  // Manual resizing only makes sense when the input is bottom-anchored (e.g.
  // the conversation page). On the home page the input is centred and dragging
  // the top grip would grow the box downward, so we disable the grip.
  const [canResize, setCanResize] = useState(true);

  const { setShouldHideSuggestions, clearMessageToSend } =
    useConversationStore();

  useEffect(() => {
    const check = () => setCanResize(isBottomAnchored());
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Callback to handle height changes and manage suggestions visibility
  const handleHeightChange = useCallback(
    (height: number) => {
      // Hide suggestions when input height exceeds the threshold
      const shouldHideChatSuggestions = height > CHAT_INPUT.HEIGHT_THRESHOLD;
      setShouldHideSuggestions(shouldHideChatSuggestions);
    },
    [setShouldHideSuggestions],
  );

  // Use the auto-resize hook with height change callback
  const {
    smartResize,
    handleGripMouseDown,
    handleGripTouchStart,
    increaseHeightForEmptyContent,
    resetManualResize,
  } = useAutoResize(chatInputRef as React.RefObject<HTMLElement | null>, {
    minHeight: 20,
    maxHeight: 400,
    onHeightChange: handleHeightChange,
    value: messageToSend ?? undefined,
    onValueApplied: clearMessageToSend, // one-shot consume (see store action)
    enableManualResize: true,
  });

  return {
    canResize,
    smartResize,
    handleGripMouseDown,
    handleGripTouchStart,
    increaseHeightForEmptyContent,
    resetManualResize,
  };
};
