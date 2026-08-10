import React from "react";
import { cn } from "#/utils/utils";

const SCROLL_EDGE_THRESHOLD_PX = 1;
const FADE_WIDTH_CLASS = "w-10";

interface MarkdownTableScrollProps {
  children: React.ReactNode;
}

interface ScrollFadeState {
  left: boolean;
  right: boolean;
}

export function readScrollFadeState(element: HTMLDivElement): ScrollFadeState {
  const { scrollLeft, scrollWidth, clientWidth } = element;
  const maxScroll = scrollWidth - clientWidth;
  const hasOverflow = maxScroll > SCROLL_EDGE_THRESHOLD_PX;

  return {
    left: hasOverflow && scrollLeft > SCROLL_EDGE_THRESHOLD_PX,
    right: hasOverflow && scrollLeft < maxScroll - SCROLL_EDGE_THRESHOLD_PX,
  };
}

export function MarkdownTableScroll({ children }: MarkdownTableScrollProps) {
  const scrollRef = React.useRef<HTMLDivElement>(null);
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

    const table = element.firstElementChild;
    if (table) {
      resizeObserver.observe(table);
    }

    return () => resizeObserver.disconnect();
  }, [updateFadeState, children]);

  return (
    <div className="relative max-w-full">
      <div
        ref={scrollRef}
        data-testid="markdown-table-scroll"
        onScroll={updateFadeState}
        className="max-w-full overflow-x-auto custom-scrollbar-always"
      >
        {children}
      </div>
      <div
        aria-hidden
        data-testid="markdown-table-scroll-fade-left"
        data-visible={fadeState.left ? "true" : "false"}
        className={cn(
          "pointer-events-none absolute inset-y-0 left-0 z-10",
          FADE_WIDTH_CLASS,
          "bg-gradient-to-r from-base to-transparent",
          "transition-opacity duration-300 ease-out motion-reduce:transition-none",
          fadeState.left ? "opacity-100" : "opacity-0",
        )}
      />
      <div
        aria-hidden
        data-testid="markdown-table-scroll-fade-right"
        data-visible={fadeState.right ? "true" : "false"}
        className={cn(
          "pointer-events-none absolute inset-y-0 right-0 z-10",
          FADE_WIDTH_CLASS,
          "bg-gradient-to-l from-base to-transparent",
          "transition-opacity duration-300 ease-out motion-reduce:transition-none",
          fadeState.right ? "opacity-100" : "opacity-0",
        )}
      />
    </div>
  );
}
