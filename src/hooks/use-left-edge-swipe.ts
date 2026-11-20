import {
  useRef,
  type TouchEvent as ReactTouchEvent,
  type TouchList as ReactTouchList,
} from "react";

/** A swipe must begin within this distance of the element's left edge. */
const EDGE_SWIPE_START_ZONE_PX = 32;
/** Rightward travel required before the gesture counts as a swipe. */
const EDGE_SWIPE_MIN_DISTANCE_PX = 56;
/** Vertical travel tolerated before the touch is treated as a scroll. */
const EDGE_SWIPE_VERTICAL_SLOP_PX = 10;
/** Horizontal travel must exceed vertical travel by this factor. */
const EDGE_SWIPE_HORIZONTAL_RATIO = 1.5;

const TEXT_ENTRY_SELECTOR =
  "input, textarea, select, [contenteditable]:not([contenteditable='false'])";
const HORIZONTAL_SCROLL_OVERFLOW = new Set(["auto", "scroll", "overlay"]);

interface SwipeStart {
  identifier: number;
  x: number;
  y: number;
}

interface UseLeftEdgeSwipeOptions {
  enabled: boolean;
  onSwipe: () => void;
}

export interface LeftEdgeSwipeHandlers {
  onTouchStart: (event: ReactTouchEvent<HTMLElement>) => void;
  onTouchMove: (event: ReactTouchEvent<HTMLElement>) => void;
  onTouchEnd: (event: ReactTouchEvent<HTMLElement>) => void;
  onTouchCancel: () => void;
}

function findTouch(list: ReactTouchList, identifier: number) {
  for (let index = 0; index < list.length; index += 1) {
    const touch = list[index];
    if (touch.identifier === identifier) return touch;
  }
  return null;
}

/** Whether a touch starting at `target` belongs to a horizontal scroller. */
function isInHorizontalScroller(target: Element, boundary: Element) {
  for (
    let element: Element | null = target;
    element && element !== boundary;
    element = element.parentElement
  ) {
    if (
      element.scrollWidth > element.clientWidth &&
      HORIZONTAL_SCROLL_OVERFLOW.has(window.getComputedStyle(element).overflowX)
    ) {
      return true;
    }
  }
  return false;
}

function isRightwardSwipe(deltaX: number, deltaY: number) {
  return (
    deltaX >= EDGE_SWIPE_MIN_DISTANCE_PX &&
    deltaX >= Math.abs(deltaY) * EDGE_SWIPE_HORIZONTAL_RATIO
  );
}

/**
 * Touch handlers that call `onSwipe` for a deliberate rightward swipe that
 * starts at the element's left edge. Vertical scrolls, short or diagonal
 * swipes, multi-touch, text-entry controls and horizontal scrollers keep their
 * native behaviour. Spread the result onto the element that owns the edge.
 */
export function useLeftEdgeSwipe({
  enabled,
  onSwipe,
}: UseLeftEdgeSwipeOptions): LeftEdgeSwipeHandlers {
  const startRef = useRef<SwipeStart | null>(null);

  const onTouchStart = (event: ReactTouchEvent<HTMLElement>) => {
    startRef.current = null;
    if (!enabled || event.touches.length !== 1) return;

    const container = event.currentTarget;
    const { target } = event;
    // Touches from portaled content bubble through React but sit elsewhere.
    if (!(target instanceof Element) || !container.contains(target)) return;

    const touch = event.touches[0];
    const edgeOffset = touch.clientX - container.getBoundingClientRect().left;
    if (edgeOffset > EDGE_SWIPE_START_ZONE_PX) return;
    if (
      target.closest(TEXT_ENTRY_SELECTOR) ||
      isInHorizontalScroller(target, container)
    ) {
      return;
    }

    startRef.current = {
      identifier: touch.identifier,
      x: touch.clientX,
      y: touch.clientY,
    };
  };

  const onTouchMove = (event: ReactTouchEvent<HTMLElement>) => {
    const start = startRef.current;
    if (!start) return;

    const touch =
      event.touches.length === 1
        ? findTouch(event.touches, start.identifier)
        : null;
    if (!touch) {
      startRef.current = null;
      return;
    }

    const deltaX = touch.clientX - start.x;
    const deltaY = touch.clientY - start.y;
    if (
      Math.abs(deltaY) > EDGE_SWIPE_VERTICAL_SLOP_PX &&
      Math.abs(deltaY) >= Math.abs(deltaX)
    ) {
      // The browser is scrolling; leave the rest of this touch alone.
      startRef.current = null;
      return;
    }

    if (isRightwardSwipe(deltaX, deltaY)) {
      startRef.current = null;
      onSwipe();
    }
  };

  const onTouchEnd = (event: ReactTouchEvent<HTMLElement>) => {
    const start = startRef.current;
    startRef.current = null;
    if (!start) return;

    const touch = findTouch(event.changedTouches, start.identifier);
    if (
      touch &&
      isRightwardSwipe(touch.clientX - start.x, touch.clientY - start.y)
    ) {
      onSwipe();
    }
  };

  const onTouchCancel = () => {
    startRef.current = null;
  };

  return { onTouchStart, onTouchMove, onTouchEnd, onTouchCancel };
}
