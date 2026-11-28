import {
  RefObject,
  useState,
  useCallback,
  useRef,
  useEffect,
  useLayoutEffect,
} from "react";

/**
 * Distance from the bottom (px) still treated as "pinned to the bottom".
 *
 * Shared with the virtualized list's anchor correction: the two "am I pinned"
 * checks have to agree, or the anchor correction and autoscroll disagree about
 * the same frame.
 */
export const BOTTOM_THRESHOLD_PX = 20;

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

  // `autoscroll` for code outside React's render cycle (the streamed-reply
  // follower and the resize observer below). Scroll handlers write it
  // synchronously, since the state lags a render behind the scroll.
  const autoscrollRef = useRef(autoscroll);
  useLayoutEffect(() => {
    autoscrollRef.current = autoscroll;
  }, [autoscroll]);

  // Check if the scroll position is at the bottom
  const isAtBottom = useCallback((element: HTMLElement): boolean => {
    const bottomPosition = element.scrollTop + element.clientHeight;
    return bottomPosition >= element.scrollHeight - BOTTOM_THRESHOLD_PX;
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
        autoscrollRef.current = false;
        setAutoscroll(false);
      }

      // Turn on autoscroll when scrolled to the bottom
      if (isCurrentlyAtBottom) {
        autoscrollRef.current = true;
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
        autoscrollRef.current = true;
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

  // A resize keeps the old scrollTop, so a transcript that reflows taller (a
  // narrower window, a rotated phone, the side panel opening) or whose content
  // grows in place would leave a pinned reader above the new bottom. Re-pin
  // on any size change of the container or its content while autoscroll is on;
  // a reader who scrolled up keeps their position. The pin comes from the
  // reader's scrolling, not from the geometry after the change: by then a
  // reflow has already moved a pinned reader off the bottom, and a scroll
  // event dispatched after the content grew reads a stale gap.
  useEffect(() => {
    const dom = scrollRef.current;
    if (!dom || typeof ResizeObserver === "undefined") return undefined;

    const repin = () => {
      if (!autoscrollRef.current) return;
      const bottom = dom.scrollHeight - dom.clientHeight;
      if (dom.scrollTop < bottom - 1) dom.scrollTop = bottom;
    };
    const resize = new ResizeObserver(repin);
    resize.observe(dom);
    // Observing a node tracks only its own box, so observe the content too.
    // Rows render inside `display: contents` wrappers, which have no box to
    // report, so look through them to the first boxed descendants, and keep
    // that set current as rows mount and unmount.
    const observed = new Set<Element>();
    const wrappers = new Set<Element>();
    const reconcile = () => {
      const boxes = new Set<Element>();
      wrappers.clear();
      const collect = (parent: Element) => {
        Array.from(parent.children).forEach((child) => {
          if (getComputedStyle(child).display === "contents") {
            wrappers.add(child);
            collect(child);
          } else {
            boxes.add(child);
          }
        });
      };
      collect(dom);
      boxes.forEach((box) => {
        if (!observed.has(box)) {
          resize.observe(box);
          observed.add(box);
        }
      });
      observed.forEach((box) => {
        if (!boxes.has(box)) {
          resize.unobserve(box);
          observed.delete(box);
        }
      });
    };
    reconcile();
    // Watch the whole subtree, since rows mount inside the wrappers, but only
    // reconcile for the container's or a wrapper's own children: a streamed
    // reply adds nodes inside a row on every frame.
    const mutation =
      typeof MutationObserver === "undefined"
        ? undefined
        : new MutationObserver((records) => {
            if (
              records.some(
                ({ target }) =>
                  target === dom || wrappers.has(target as Element),
              )
            ) {
              reconcile();
            }
          });
    mutation?.observe(dom, { childList: true, subtree: true });

    return () => {
      resize.disconnect();
      mutation?.disconnect();
    };
  }, [scrollRef]);

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
