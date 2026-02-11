import {
  RefObject,
  useState,
  useCallback,
  useRef,
  useLayoutEffect,
} from "react";

export function useScrollToBottom(scrollRef: RefObject<HTMLDivElement | null>) {
  // Track whether we should auto-scroll to the bottom when content changes
  const [autoscroll, setAutoscroll] = useState(true);

  // Track whether the user is currently at the bottom of the scroll area
  const [hitBottom, setHitBottom] = useState(true);

  // Store previous scroll position to detect scroll direction
  const prevScrollTopRef = useRef<number>(0);

  // Track last scrollHeight to avoid redundant scroll operations.
  // Without this guard, the useLayoutEffect below would set scrollTop
  // on every render (including resize-triggered re-renders), even when
  // the content hasn't changed.
  const lastScrollHeightRef = useRef<number>(0);

  // Check if the scroll position is at the bottom
  const isAtBottom = useCallback((element: HTMLElement): boolean => {
    // Use a fixed 20px buffer
    const bottomThreshold = 20;
    const bottomPosition = element.scrollTop + element.clientHeight;
    return bottomPosition >= element.scrollHeight - bottomThreshold;
  }, []);

  // Handle scroll events
  const onChatBodyScroll = useCallback(
    (e: HTMLElement) => {
      const isCurrentlyAtBottom = isAtBottom(e);
      setHitBottom(isCurrentlyAtBottom);

      // Get current scroll position
      const currentScrollTop = e.scrollTop;

      // Detect scroll direction
      const isScrollingUp = currentScrollTop < prevScrollTopRef.current;

      // Update previous scroll position for next comparison
      prevScrollTopRef.current = currentScrollTop;

      // Turn off autoscroll only when scrolling up
      if (isScrollingUp) {
        setAutoscroll(false);
      }

      // Turn on autoscroll when scrolled to the bottom
      if (isCurrentlyAtBottom) {
        setAutoscroll(true);
      }
    },
    [isAtBottom],
  );

  // Scroll to bottom function with animation
  const scrollDomToBottom = useCallback(() => {
    const dom = scrollRef.current;
    if (dom) {
      requestAnimationFrame(() => {
        // Set autoscroll to true when manually scrolling to bottom
        setAutoscroll(true);
        setHitBottom(true);

        dom.scrollTop = dom.scrollHeight;
      });
    }
  }, [scrollRef]);

  // Auto-scroll effect that runs when content changes
  // Use useLayoutEffect to scroll after DOM updates but before paint
  useLayoutEffect(() => {
    if (autoscroll) {
      const dom = scrollRef.current;
      if (dom) {
        const { scrollHeight } = dom;
        // Only scroll when content has actually changed (scrollHeight differs).
        // This prevents redundant DOM writes during resize-triggered re-renders,
        // where the component re-renders but the message content is unchanged.
        if (scrollHeight !== lastScrollHeightRef.current) {
          lastScrollHeightRef.current = scrollHeight;
          dom.scrollTop = scrollHeight;
        }
      }
    }
  }); // No dependency array - runs after every render, but guards against unchanged content

  return {
    scrollRef,
    autoScroll: autoscroll,
    setAutoScroll: setAutoscroll,
    scrollDomToBottom,
    hitBottom,
    setHitBottom,
    onChatBodyScroll,
  };
}
