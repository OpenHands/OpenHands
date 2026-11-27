import type {
  RefObject,
  MouseEvent as ReactMouseEvent,
  TouchEvent as ReactTouchEvent,
} from "react";
import { useCallback, useEffect, useRef, useState } from "react";

const DRAG_COMMIT_PX = 2;
const DEFAULT_MIN_HEIGHT = 120;
const DEFAULT_MAX_HEIGHT = 400;

interface UsePromptTextareaResizeOptions {
  minHeight?: number;
  maxHeight?: number;
  /** Re-measure overflow when the field value changes without changing the dragged height. */
  contentKey?: unknown;
}

function styleHeightPx(element: HTMLElement, fallback: number): number {
  const height = Number.parseFloat(element.style.height);
  return Number.isFinite(height) ? height : fallback;
}

export function usePromptTextareaResize<T extends HTMLElement>(
  elementRef: RefObject<T | null>,
  {
    minHeight = DEFAULT_MIN_HEIGHT,
    maxHeight = DEFAULT_MAX_HEIGHT,
    contentKey,
  }: UsePromptTextareaResizeOptions = {},
) {
  const [isGripDragging, setIsGripDragging] = useState(false);
  const gripRef = useRef<HTMLDivElement | null>(null);

  const applyHeight = useCallback(
    (height: number) => {
      const element = elementRef.current;
      if (!element) return;
      const nextHeight = Math.max(minHeight, Math.min(maxHeight, height));
      element.style.height = `${nextHeight}px`;
      element.style.overflowY =
        element.scrollHeight > nextHeight ? "auto" : "hidden";
    },
    [elementRef, maxHeight, minHeight],
  );

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;
    applyHeight(styleHeightPx(element, minHeight));
  }, [applyHeight, contentKey, elementRef, minHeight]);

  const startDrag = useCallback(
    (startY: number) => {
      const element = elementRef.current;
      if (!element) return;

      const startHeight = styleHeightPx(element, minHeight);
      let dragCommitted = false;

      const handleDragMove = (moveEvent: MouseEvent | TouchEvent) => {
        moveEvent.preventDefault();

        const clientY =
          "touches" in moveEvent && moveEvent.touches.length > 0
            ? moveEvent.touches[0].clientY
            : (moveEvent as MouseEvent).clientY;

        if (!dragCommitted) {
          if (Math.abs(clientY - startY) < DRAG_COMMIT_PX) return;
          dragCommitted = true;
          setIsGripDragging(true);
        }

        applyHeight(startHeight + (clientY - startY));
      };

      const handleDragEnd = () => {
        document.removeEventListener("mousemove", handleDragMove);
        document.removeEventListener("mouseup", handleDragEnd);
        document.removeEventListener("touchmove", handleDragMove);
        document.removeEventListener("touchend", handleDragEnd);
        setIsGripDragging(false);
      };

      document.addEventListener("mousemove", handleDragMove);
      document.addEventListener("mouseup", handleDragEnd);
      document.addEventListener("touchmove", handleDragMove, {
        passive: false,
      });
      document.addEventListener("touchend", handleDragEnd);
    },
    [applyHeight, elementRef, minHeight],
  );

  const handleGripMouseDown = useCallback(
    (event: ReactMouseEvent) => {
      event.preventDefault();
      startDrag(event.clientY);
    },
    [startDrag],
  );

  const handleGripTouchStart = useCallback(
    (event: ReactTouchEvent) => {
      event.preventDefault();
      startDrag(event.touches[0].clientY);
    },
    [startDrag],
  );

  return {
    gripRef,
    isGripDragging,
    handleGripMouseDown,
    handleGripTouchStart,
  };
}
