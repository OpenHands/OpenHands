import type {
  RefObject,
  MouseEvent as ReactMouseEvent,
  TouchEvent as ReactTouchEvent,
} from "react";
import { EPS } from "#/utils/constants";
import { isMobileDevice } from "#/utils/utils";

/** Movement required before a pointer gesture is treated as a resize drag. */
const DRAG_COMMIT_PX = 2;

/**
 * Distance (px) from the viewport bottom within which we consider the chat
 * input to be "bottom-anchored" by its parent's flexbox. When the input is
 * bottom-anchored (e.g. the conversation page), dragging the top grip makes
 * the box grow upward — the expected behaviour. When it is NOT
 * bottom-anchored (e.g. the home page where the input sits in the centre of
 * the screen), the same drag would make the box grow downward, which is
 * confusing, so we disable manual resizing entirely in that case.
 */
const BOTTOM_ANCHORED_THRESHOLD = 120;

/**
 * Determine whether the chat input is anchored to the bottom of the viewport
 * by its parent layout (e.g. the conversation page uses a flex column with a
 * growing scroll area above the input). When true, dragging the top grip
 * makes the box grow upward, which is the expected behaviour. When false,
 * the drag would grow the box downward, so we disable it.
 */
export const isBottomAnchored = (): boolean => {
  const grip = document.getElementById("resize-grip");
  const wrapper = grip?.parentElement ?? null;
  if (!wrapper) return true; // fall back to allowing the drag
  const rect = wrapper.getBoundingClientRect();
  return rect.bottom >= window.innerHeight - BOTTOM_ANCHORED_THRESHOLD;
};

// Drag handling hook
interface UseDragResizeOptions {
  elementRef: RefObject<HTMLElement | null>;
  minHeight: number;
  maxHeight: number;
  onGripDragStart?: () => void;
  onGripDragEnd?: () => void;
  onHeightChange?: (height: number) => void;
  onReachedMinHeight?: () => void; // Notify when user drags to minimum height
}

export const useDragResize = ({
  elementRef,
  minHeight,
  maxHeight,
  onGripDragStart,
  onGripDragEnd,
  onHeightChange,
  onReachedMinHeight,
}: UseDragResizeOptions) => {
  const getClientY = (event: MouseEvent | TouchEvent): number => {
    if ("touches" in event && event.touches.length > 0) {
      return event.touches[0].clientY;
    }
    return (event as MouseEvent).clientY;
  };

  // Setup event listeners for mobile devices
  const setupMobileEventListeners = (
    handleDragMove: (event: MouseEvent | TouchEvent) => void,
    handleDragEnd: () => void,
  ) => {
    const resizeGrip = document.getElementById("resize-grip");

    if (!resizeGrip) {
      return;
    }

    resizeGrip.addEventListener("touchmove", handleDragMove, {
      passive: false,
      capture: true,
    });
    resizeGrip.addEventListener("touchend", handleDragEnd, {
      capture: true,
    });
  };

  // Setup event listeners for desktop devices
  const setupDesktopEventListeners = (
    handleDragMove: (event: MouseEvent | TouchEvent) => void,
    handleDragEnd: () => void,
  ) => {
    document.addEventListener("mousemove", handleDragMove);
    document.addEventListener("mouseup", handleDragEnd);
  };

  // Start drag operation
  const startDrag = (startY: number) => {
    const isMobile = isMobileDevice();
    const startHeight = elementRef.current?.offsetHeight || minHeight;
    let dragCommitted = false;

    const handleDragMove = (moveEvent: MouseEvent | TouchEvent) => {
      moveEvent.preventDefault();

      if (!dragCommitted) {
        const y = getClientY(moveEvent);
        if (Math.abs(y - startY) < DRAG_COMMIT_PX) {
          return;
        }
        dragCommitted = true;
        onGripDragStart?.();
      }

      const deltaY = getClientY(moveEvent) - startY;
      // Invert deltaY so moving up increases height and moving down decreases height
      const newHeight = Math.max(
        minHeight,
        Math.min(maxHeight, startHeight - deltaY),
      );

      const element = elementRef.current;

      if (!element) {
        return;
      }

      element.style.height = `${newHeight}px`;

      // Check if content exceeds the new height to determine scrollbar visibility
      const contentHeight = element.scrollHeight;
      const shouldShowScrollbar =
        contentHeight > newHeight || newHeight >= maxHeight;
      element.style.overflowY = shouldShowScrollbar ? "auto" : "hidden";

      // Call the height change callback if provided
      if (onHeightChange) {
        onHeightChange(newHeight);
      }

      // Notify when dragged to minimum height to clear manual mode
      if (onReachedMinHeight && Math.abs(newHeight - minHeight) <= EPS) {
        onReachedMinHeight();
      }
    };

    const handleDragEnd = () => {
      if (dragCommitted) {
        onGripDragEnd?.();
      }

      if (isMobile) {
        const resizeGrip = document.getElementById("resize-grip");
        if (!resizeGrip) {
          return;
        }

        // Remove both mouse and touch event listeners
        resizeGrip.removeEventListener("mousemove", handleDragMove);
        resizeGrip.removeEventListener("mouseup", handleDragEnd);
        resizeGrip.removeEventListener("touchmove", handleDragMove);
        resizeGrip.removeEventListener("touchend", handleDragEnd);
      } else {
        document.removeEventListener("mousemove", handleDragMove);
        document.removeEventListener("mouseup", handleDragEnd);
        document.removeEventListener("touchmove", handleDragMove);
        document.removeEventListener("touchend", handleDragEnd);
      }
    };

    // Setup event listeners based on device type
    if (isMobile) {
      setupMobileEventListeners(handleDragMove, handleDragEnd);
    } else {
      setupDesktopEventListeners(handleDragMove, handleDragEnd);
    }
  };

  // Handle mouse down on grip for manual resizing
  const handleGripMouseDown = (e: ReactMouseEvent) => {
    e.preventDefault();
    // Only allow manual resizing when the input is bottom-anchored (e.g. the
    // conversation page). On the home page the input is centred and dragging
    // the top grip would grow the box downward, which is confusing.
    if (!isBottomAnchored()) return;
    startDrag(e.clientY);
  };

  // Handle touch start on grip for manual resizing
  const handleGripTouchStart = (e: ReactTouchEvent) => {
    e.preventDefault();
    if (!isBottomAnchored()) return;
    startDrag(e.touches[0].clientY);
  };

  return {
    handleGripMouseDown,
    handleGripTouchStart,
  };
};
