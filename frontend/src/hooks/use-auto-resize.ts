import { useCallback, useEffect, RefObject, useRef } from "react";
import { IMessageToSend } from "#/stores/conversation-store";
import { EPS } from "#/utils/constants";
import { getStyleHeightPx, setStyleHeightPx } from "#/utils/utils";
import { useDragResize } from "./use-drag-resize";

// Constants
const DEFAULT_MIN_HEIGHT = 20;
const DEFAULT_MAX_HEIGHT = 120;
const HEIGHT_INCREMENT = 20;
const MANUAL_OVERSIZE_THRESHOLD = 50;

// Manual height tracking utilities
const useManualHeight = () => {
  const hasUserResizedRef = useRef(false);
  const manualHeightRef = useRef<number | null>(null);
  return { hasUserResizedRef, manualHeightRef };
};

interface UseAutoResizeOptions {
  minHeight?: number;
  maxHeight?: number;
  enableManualResize?: boolean;
  onGripDragStart?: () => void;
  onGripDragEnd?: () => void;
  onHeightChange?: (height: number) => void; // New callback for height changes
  value?: IMessageToSend;
}

interface UseAutoResizeReturn {
  smartResize: () => void;
  handleGripMouseDown: (e: React.MouseEvent) => void;
  handleGripTouchStart: (e: React.TouchEvent) => void;
  increaseHeightForEmptyContent: () => void;
  resetManualResize: () => void;
}

// Height management utilities
interface HeightConstraints {
  minHeight: number;
  maxHeight: number;
}

interface HeightMeasurements {
  currentHeight: number;
  currentStyleHeight: number;
  contentHeight: number;
}

interface ResizeStrategy {
  finalHeight: number;
  overflowY: "hidden" | "auto";
}

const applyHeightToElement = (
  element: HTMLElement,
  height: number,
  constraints: HeightConstraints,
): number => {
  const { minHeight, maxHeight } = constraints;
  const finalHeight = Math.max(minHeight, Math.min(height, maxHeight));

  element.style.setProperty("height", `${finalHeight}px`);
  element.style.setProperty(
    "overflow-y",
    finalHeight >= maxHeight ? "auto" : "hidden",
  );

  return finalHeight;
};

const isManuallyOversized = (
  currentHeight: number,
  contentHeight: number,
  threshold = MANUAL_OVERSIZE_THRESHOLD,
): boolean => currentHeight > contentHeight + threshold;

const measureElementHeights = (
  element: HTMLElement,
  minHeight: number,
): HeightMeasurements => {
  // Use the previous explicit style height as the "current" for restore, not offsetHeight
  const currentStyleHeight = getStyleHeightPx(element, minHeight);
  const currentHeight = currentStyleHeight;

  // Temporarily reset to measure content
  element.style.setProperty("height", "auto");
  const contentHeight = element.scrollHeight;

  // Restore height
  setStyleHeightPx(element, currentStyleHeight);

  return {
    currentHeight,
    currentStyleHeight,
    contentHeight,
  };
};

const applyResizeStrategy = (
  element: HTMLElement,
  strategy: ResizeStrategy,
): void => {
  const { finalHeight, overflowY } = strategy;
  const elementRef = element;
  elementRef.style.height = `${finalHeight}px`;
  elementRef.style.overflowY = overflowY;
};

const executeHeightCallback = (
  height: number,
  onHeightChange?: (height: number) => void,
): void => {
  if (onHeightChange) {
    onHeightChange(height);
  }
};

// DOM manipulation utilities
export const useAutoResize = (
  elementRef: RefObject<HTMLElement | null>,
  options: UseAutoResizeOptions = {},
): UseAutoResizeReturn => {
  const pendingSmartRef = useRef<number | null>(null);

  const {
    minHeight = DEFAULT_MIN_HEIGHT,
    maxHeight = DEFAULT_MAX_HEIGHT,
    enableManualResize = false,
    value,
    onGripDragStart,
    onGripDragEnd,
    onHeightChange,
  } = options;

  const constraints: HeightConstraints = { minHeight, maxHeight };
  const { hasUserResizedRef, manualHeightRef } = useManualHeight();

  const resetManualResize = () => {
    hasUserResizedRef.current = false;
    manualHeightRef.current = null;
  };

  // Wrap onHeightChange to track manual height during drag
  const handleExternalHeightChange = useCallback(
    (elementHeight: number) => {
      onHeightChange?.(elementHeight);
      if (hasUserResizedRef.current) {
        manualHeightRef.current = elementHeight;
      }
    },
    [onHeightChange],
  );

  // Handle drag start - set manual mode flag
  const handleDragStart = useCallback(() => {
    hasUserResizedRef.current = true;
    onGripDragStart?.();
  }, [onGripDragStart]);

  // Handle drag end - clear manual mode if at minimum height
  const handleDragEnd = useCallback(() => {
    const textareaElement = elementRef.current;
    if (textareaElement) {
      const currentHeight = getStyleHeightPx(textareaElement, minHeight);
      if (Math.abs(currentHeight - minHeight) <= EPS) {
        hasUserResizedRef.current = false;
        manualHeightRef.current = null;
      }
    }
    onGripDragEnd?.();
  }, [minHeight, onGripDragEnd]);

  // Use the drag resize hook for manual resizing functionality
  const { handleGripMouseDown, handleGripTouchStart } = useDragResize({
    elementRef,
    minHeight,
    maxHeight,
    onGripDragStart: enableManualResize ? handleDragStart : undefined,
    onGripDragEnd: enableManualResize ? handleDragEnd : undefined,
    onHeightChange: handleExternalHeightChange,
  });

  // Handle content that fits within current height
  const handleContentFitsInCurrentHeight = useCallback(
    (
      element: HTMLElement,
      currentHeight: number,
      contentHeight: number,
    ): void => {
      // If user manually resized and we're above min height, preserve their chosen height
      if (hasUserResizedRef.current && currentHeight > minHeight) {
        applyResizeStrategy(element, {
          finalHeight: currentHeight,
          overflowY: "hidden",
        });
        executeHeightCallback(currentHeight, onHeightChange);
        return;
      }

      // Otherwise allow shrinking towards content (respect minHeight)
      const finalHeight = Math.max(contentHeight, minHeight);
      applyResizeStrategy(element, {
        finalHeight,
        overflowY: "hidden",
      });
      executeHeightCallback(finalHeight, onHeightChange);
    },
    [minHeight, onHeightChange],
  );

  // Handle content that exceeds current height but within max height
  const handleContentExceedsCurrentHeight = useCallback(
    (
      element: HTMLElement,
      currentHeight: number,
      contentHeight: number,
    ): void => {
      // Grow unless the element is manually oversized beyond content significantly
      if (!isManuallyOversized(currentHeight, contentHeight)) {
        const finalHeight = Math.max(contentHeight, minHeight);
        applyResizeStrategy(element, {
          finalHeight,
          overflowY: "hidden",
        });
        executeHeightCallback(finalHeight, onHeightChange);
        return;
      }

      // Keep manual height and allow scrolling as needed
      applyResizeStrategy(element, {
        finalHeight: currentHeight,
        overflowY: "auto",
      });
      executeHeightCallback(currentHeight, onHeightChange);
    },
    [minHeight, onHeightChange],
  );

  // Handle content that exceeds max height
  const handleContentExceedsMaxHeight = useCallback(
    (element: HTMLElement) => {
      applyResizeStrategy(element, {
        finalHeight: maxHeight,
        overflowY: "auto",
      });
      executeHeightCallback(maxHeight, onHeightChange);
    },
    [maxHeight, onHeightChange],
  );

  // Debounced smartResize body
  const smartResizeBody = useCallback(() => {
    const element = elementRef.current;
    if (!element) return;

    const textIsEmpty = (element.textContent ?? "").trim().length === 0;

    // If empty content and we have a manual height above min, preserve it
    if (
      textIsEmpty &&
      hasUserResizedRef.current &&
      manualHeightRef.current &&
      manualHeightRef.current > minHeight + EPS
    ) {
      setStyleHeightPx(element, manualHeightRef.current);
      element.style.overflowY = "hidden";
      executeHeightCallback(manualHeightRef.current, onHeightChange);
      return;
    }

    // Measure element heights
    const measurements = measureElementHeights(element, minHeight);
    const { currentHeight, contentHeight } = measurements;

    // If content fits within current height
    if (contentHeight <= currentHeight) {
      handleContentFitsInCurrentHeight(element, currentHeight, contentHeight);
      return;
    }

    // If content exceeds current height but within max
    if (contentHeight <= maxHeight) {
      handleContentExceedsCurrentHeight(element, currentHeight, contentHeight);
      return;
    }

    // Content exceeds max height
    handleContentExceedsMaxHeight(element);
  }, [
    elementRef,
    minHeight,
    maxHeight,
    onHeightChange,
    handleContentFitsInCurrentHeight,
    handleContentExceedsCurrentHeight,
    handleContentExceedsMaxHeight,
  ]);

  // rAF-debounced smartResize wrapper to collapse bursts
  const smartResize = useCallback(() => {
    if (pendingSmartRef.current) cancelAnimationFrame(pendingSmartRef.current);
    pendingSmartRef.current = requestAnimationFrame(() => {
      pendingSmartRef.current = null;
      smartResizeBody();
    });
  }, [smartResizeBody]);

  // Function to increase height when content is empty
  const increaseHeightForEmptyContent = () => {
    const element = elementRef.current;
    if (!element) return;

    const currentHeight = getStyleHeightPx(element, minHeight);
    const newHeight = Math.min(currentHeight + HEIGHT_INCREMENT, maxHeight);

    if (newHeight > currentHeight) {
      const finalHeight = applyHeightToElement(element, newHeight, constraints);

      // Execute height change callback
      executeHeightCallback(finalHeight, onHeightChange);

      // Set manual mode for Shift+Enter height increases
      hasUserResizedRef.current = true;
      manualHeightRef.current = finalHeight;
    }
  };

  // Update content and resize when value prop changes
  useEffect(() => {
    const element = elementRef.current;
    if (element && value !== undefined) {
      element.textContent = value.text;
      smartResize();
    }
  }, [value, smartResize]);

  // Initialize auto-resize on mount
  useEffect(() => {
    smartResize();
  }, [smartResize]);

  return {
    smartResize,
    handleGripMouseDown,
    handleGripTouchStart,
    increaseHeightForEmptyContent,
    resetManualResize,
  };
};
