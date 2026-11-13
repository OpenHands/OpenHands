import {
  RefObject,
  useState,
  useCallback,
  useRef,
  useEffect,
  useLayoutEffect,
} from "react";

/**
 * @param followContent Content is growing in place (a streamed reply), so keep
 *   the bottom in view frame by frame while the user is pinned to it.
 */
export function useScrollToBottom(
  scrollRef: RefObject<HTMLDivElement | null>,
  followContent = false,
) {
  // Track whether the user is currently near the bottom of the scroll area.
  // Used by consumers to decide whether to scroll when new UI elements appear.
  // NOT used for automatic content-following.
  const [autoscroll, setAutoscroll] = useState(true);

  // Track whether the user is currently at the bottom of the scroll area
  const [hitBottom, setHitBottom] = useState(true);

  // Store previous scroll position to detect scroll direction
  const prevScrollTopRef = useRef<number>(0);

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

  // Scroll to bottom on manual click only
  const scrollDomToBottom = useCallback(() => {
    const dom = scrollRef.current;
    if (dom) {
      requestAnimationFrame(() => {
        setAutoscroll(true);
        setHitBottom(true);

        dom.scrollTop = dom.scrollHeight;
      });
    }
  }, [scrollRef]);

  // A streamed reply grows inside a single event, so nothing re-renders to
  // scroll it into view. The user scrolling up is read from the element each
  // frame: the `autoscroll` state that records it lags a render behind, and
  // following until then would yank them back down.
  const autoscrollRef = useRef(autoscroll);
  useLayoutEffect(() => {
    autoscrollRef.current = autoscroll;
  }, [autoscroll]);

  useEffect(() => {
    const dom = scrollRef.current;
    if (!followContent || !dom || typeof requestAnimationFrame !== "function") {
      return undefined;
    }

    let lastTop = dom.scrollTop;
    let frame = 0;
    const follow = () => {
      const bottom = dom.scrollHeight - dom.clientHeight;
      if (autoscrollRef.current) {
        // Content shrinking also lowers scrollTop, but leaves it at the bottom.
        if (dom.scrollTop < lastTop - 1 && dom.scrollTop < bottom - 1) {
          autoscrollRef.current = false;
          setAutoscroll(false);
        } else if (dom.scrollTop < bottom - 1) {
          dom.scrollTop = bottom;
        }
      }
      lastTop = dom.scrollTop;
      frame = requestAnimationFrame(follow);
    };

    frame = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(frame);
  }, [followContent, scrollRef]);

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
