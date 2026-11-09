import React from "react";
import { cn } from "#/utils/utils";
import {
  readScrollFadeState,
  type ScrollFadeState,
} from "#/utils/scroll-fade-state";

export { readScrollFadeState };

const FADE_WIDTH_CLASS = "w-10";

interface MarkdownTableScrollProps {
  children: React.ReactNode;
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
      {/* The fades blend into whatever surface the table sits on, which differs
          between the chat stream and cards like the markdown artifact preview.
          Set `--oh-scroll-fade-from` on any ancestor to match another surface.
          The shared from-scroll-fade token preserves that ancestor override
          and falls back to the Canvas base surface. */}
      <div
        aria-hidden
        data-testid="markdown-table-scroll-fade-left"
        data-visible={fadeState.left ? "true" : "false"}
        className={cn(
          "pointer-events-none absolute inset-y-0 left-0 z-10",
          FADE_WIDTH_CLASS,
          "bg-gradient-to-r from-scroll-fade to-transparent",
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
          "bg-gradient-to-l from-scroll-fade to-transparent",
          "transition-opacity duration-300 ease-out motion-reduce:transition-none",
          fadeState.right ? "opacity-100" : "opacity-0",
        )}
      />
    </div>
  );
}
