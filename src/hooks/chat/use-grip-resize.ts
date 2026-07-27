import type { RefObject, MouseEvent } from "react";
import { useRef, useState, useCallback } from "react";
import { useAutoResize } from "#/hooks/use-auto-resize";
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
  const [isGripVisible, setIsGripVisible] = useState(false);
  const [isGripDragging, setIsGripDragging] = useState(false);

  const { setShouldHideSuggestions, clearMessageToSend } =
    useConversationStore();

  const gripRef = useRef<HTMLDivElement | null>(null);
  /** After a real resize drag, swallow the synthetic click so it doesn't toggle `isGripVisible`. */
  const suppressNextTopEdgeClickRef = useRef(false);

  const handleGripDragStart = useCallback(() => {
    setIsGripDragging(true);
  }, []);

  const handleGripDragEnd = useCallback(() => {
    setIsGripDragging(false);
    suppressNextTopEdgeClickRef.current = true;
  }, []);

  // Handle click on top edge area to toggle grip visibility
  const handleTopEdgeClick = useCallback((e: MouseEvent) => {
    if (suppressNextTopEdgeClickRef.current) {
      suppressNextTopEdgeClickRef.current = false;
      e.preventDefault();
      e.stopPropagation();
      return;
    }
    e.stopPropagation();
    setIsGripVisible((prev) => !prev);
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
    onGripDragStart: handleGripDragStart,
    onGripDragEnd: handleGripDragEnd,
    value: messageToSend ?? undefined,
    onValueApplied: clearMessageToSend, // one-shot consume (see store action)
    enableManualResize: true,
  });

  return {
    gripRef,
    isGripVisible,
    isGripDragging,
    handleTopEdgeClick,
    smartResize,
    handleGripMouseDown,
    handleGripTouchStart,
    increaseHeightForEmptyContent,
    resetManualResize,
  };
};
