import { RefObject } from "react";
import { EPS } from "#/utils/constants";
import { isMobileDevice } from "#/utils/utils";

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

  // Create drag move handler
  const createDragMoveHandler = (startY: number, startHeight: number) => {
    const handleDragMove = (moveEvent: MouseEvent | TouchEvent) => {
      moveEvent.preventDefault();

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
    return handleDragMove;
  };

  // Create drag end handler
  const createDragEndHandler = (
    isMobile: boolean,
    handleDragMove: (event: MouseEvent | TouchEvent) => void,
  ) => {
    const handleDragEnd = () => {
      // Call optional drag end callback
      onGripDragEnd?.();

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
    return handleDragEnd;
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
    // Call optional drag start callback
    onGripDragStart?.();

    const isMobile = isMobileDevice();
    const startHeight = elementRef.current?.offsetHeight || minHeight;

    // Create event handlers
    const handleDragMove = createDragMoveHandler(startY, startHeight);
    const handleDragEnd = createDragEndHandler(isMobile, handleDragMove);

    // Setup event listeners based on device type
    if (isMobile) {
      setupMobileEventListeners(handleDragMove, handleDragEnd);
    } else {
      setupDesktopEventListeners(handleDragMove, handleDragEnd);
    }
  };

  // Handle mouse down on grip for manual resizing
  const handleGripMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    startDrag(e.clientY);
  };

  // Handle touch start on grip for manual resizing
  const handleGripTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    startDrag(e.touches[0].clientY);
  };

  return {
    handleGripMouseDown,
    handleGripTouchStart,
  };
};
