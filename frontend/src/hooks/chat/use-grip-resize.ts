import { useRef, useState, useCallback } from "react";
import { useAutoResize } from "#/hooks/use-auto-resize";
import { CHAT_INPUT } from "#/utils/constants";
import {
  IMessageToSend,
  useConversationStore,
} from "#/stores/conversation-store";

// ---------------------------------------------------------------------------
// WHY A SELECTOR HERE
//
// Using useConversationStore() (full store, no selector) would cause the hook
// owner to re-render on *every* Zustand set() call — including the redundant
// setShouldHideSuggestions(false) calls that used to fire on every keystroke.
// Selecting only the one action we need keeps this hook's owner stable.
// ---------------------------------------------------------------------------

/**
 * Hook for managing grip resize functionality
 */
export const useGripResize = (
  chatInputRef: React.RefObject<HTMLDivElement | null>,
  messageToSend: IMessageToSend | null,
) => {
  const [isGripVisible, setIsGripVisible] = useState(false);

  // Selector: only subscribe to the one action this hook needs so that
  // unrelated store mutations don't trigger a re-render of the hook owner.
  const setShouldHideSuggestions = useConversationStore(
    (state) => state.setShouldHideSuggestions,
  );

  // Track the last value that was committed to the store.  smartResizeBody
  // fires on every keystroke, so without this guard it would call
  // setShouldHideSuggestions(false) on every keypress even though the value
  // never changes, producing a Zustand set() → new state object → full-store
  // subscriber re-renders on every input event.
  const shouldHideRef = useRef(false);

  const gripRef = useRef<HTMLDivElement | null>(null);

  // Drag state management callbacks
  const handleDragStart = useCallback(() => {
    // Keep grip visible during drag by adding a CSS class
    if (gripRef.current) {
      gripRef.current.classList.add("opacity-100");
      gripRef.current.classList.remove("opacity-0");
    }
  }, []);

  const handleDragEnd = useCallback(() => {
    // Restore hover-based visibility
    if (gripRef.current) {
      gripRef.current.classList.remove("opacity-100");
      gripRef.current.classList.add("opacity-0");
    }
  }, []);

  // Handle click on top edge area to toggle grip visibility
  const handleTopEdgeClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsGripVisible((prev) => !prev);
  }, []);

  // Callback to handle height changes and manage suggestions visibility.
  // The guard on shouldHideRef is critical for typing performance: without it
  // every keystroke triggers setShouldHideSuggestions(false) (height is almost
  // always below the threshold while typing), which calls Zustand set() on
  // every input event and forces all full-store subscribers to re-render.
  const handleHeightChange = useCallback(
    (height: number) => {
      const shouldHide = height > CHAT_INPUT.HEIGHT_THRESHOLD;
      if (shouldHide !== shouldHideRef.current) {
        shouldHideRef.current = shouldHide;
        setShouldHideSuggestions(shouldHide);
      }
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
    onGripDragStart: handleDragStart,
    onGripDragEnd: handleDragEnd,
    value: messageToSend ?? undefined,
    enableManualResize: true,
  });

  return {
    gripRef,
    isGripVisible,
    handleTopEdgeClick,
    smartResize,
    handleGripMouseDown,
    handleGripTouchStart,
    increaseHeightForEmptyContent,
    resetManualResize,
  };
};
