import React from "react";
import { useDragScroll } from "#/hooks/use-drag-scroll";
import {
  readScrollFadeState,
  SCROLL_EDGE_THRESHOLD_PX,
  type ScrollFadeState,
} from "#/utils/scroll-fade-state";
import { cn } from "#/utils/utils";

const FADE_WIDTH_CLASS = "w-10";

interface HorizontalScrollFadeProps {
  children: React.ReactNode;
  className?: string;
  scrollClassName?: string;
  scrollTestId?: string;
  fadeTestIdPrefix?: string;
  /**
   * Lets a mouse drag the content sideways and turns a vertical wheel turn
   * into a horizontal scroll while the content overflows. Trackpads and touch
   * scroll natively either way.
   */
  draggable?: boolean;
}

export function HorizontalScrollFade({
  children,
  className,
  scrollClassName,
  scrollTestId,
  fadeTestIdPrefix = "horizontal-scroll-fade",
  draggable = false,
}: HorizontalScrollFadeProps) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const { handleMouseDown, handleClickCapture, handleDragStart } =
    useDragScroll(scrollRef);
  const [fadeState, setFadeState] = React.useState<ScrollFadeState>({
    left: false,
    right: false,
  });

  const updateFadeState = React.useCallback(() => {
    const element = scrollRef.current;
    if (!element) {
      return;
    }
    setFadeState(readScrollFadeState(element));
  }, []);

  React.useLayoutEffect(() => {
    updateFadeState();

    const element = scrollRef.current;
    if (!element) {
      return undefined;
    }

    const resizeObserver = new ResizeObserver(updateFadeState);
    resizeObserver.observe(element);

    const content = element.firstElementChild;
    if (content) {
      resizeObserver.observe(content);
    }

    return () => resizeObserver.disconnect();
  }, [updateFadeState, children]);

  React.useEffect(() => {
    const element = scrollRef.current;
    if (!draggable || !element) {
      return undefined;
    }

    const handleWheel = (event: WheelEvent) => {
      if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
      const maxScroll = element.scrollWidth - element.clientWidth;
      if (maxScroll <= SCROLL_EDGE_THRESHOLD_PX) return;
      // Clamp so a wheel step can reach either end; once the content is at an
      // edge, leave the event to scroll the page.
      const next = Math.min(
        Math.max(element.scrollLeft + event.deltaY, 0),
        maxScroll,
      );
      if (Math.abs(next - element.scrollLeft) < SCROLL_EDGE_THRESHOLD_PX) {
        return;
      }
      event.preventDefault();
      element.scrollLeft = next;
    };

    element.addEventListener("wheel", handleWheel, { passive: false });
    return () => element.removeEventListener("wheel", handleWheel);
  }, [draggable]);

  return (
    <div className={cn("relative max-w-full min-w-0", className)}>
      <div
        ref={scrollRef}
        data-testid={scrollTestId}
        onScroll={updateFadeState}
        onMouseDown={draggable ? handleMouseDown : undefined}
        onClickCapture={draggable ? handleClickCapture : undefined}
        onDragStart={draggable ? handleDragStart : undefined}
        className={cn(
          "max-w-full overflow-x-auto scrollbar-hide",
          draggable && "cursor-grab select-none active:cursor-grabbing",
          scrollClassName,
        )}
      >
        {children}
      </div>
      <div
        aria-hidden
        data-testid={`${fadeTestIdPrefix}-left`}
        data-visible={fadeState.left ? "true" : "false"}
        className={cn(
          "pointer-events-none absolute inset-y-0 left-0 z-10",
          FADE_WIDTH_CLASS,
          "bg-gradient-to-r from-[var(--oh-scroll-fade-from,var(--oh-color-base))] to-transparent",
          "transition-opacity duration-300 ease-out motion-reduce:transition-none",
          fadeState.left ? "opacity-100" : "opacity-0",
        )}
      />
      <div
        aria-hidden
        data-testid={`${fadeTestIdPrefix}-right`}
        data-visible={fadeState.right ? "true" : "false"}
        className={cn(
          "pointer-events-none absolute inset-y-0 right-0 z-10",
          FADE_WIDTH_CLASS,
          "bg-gradient-to-l from-[var(--oh-scroll-fade-from,var(--oh-color-base))] to-transparent",
          "transition-opacity duration-300 ease-out motion-reduce:transition-none",
          fadeState.right ? "opacity-100" : "opacity-0",
        )}
      />
    </div>
  );
}
