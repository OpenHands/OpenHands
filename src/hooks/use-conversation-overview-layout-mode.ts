import { useEffect, useLayoutEffect, useState, type RefObject } from "react";
import {
  getConversationOverviewLayoutMode,
  type ConversationOverviewLayoutMode,
} from "#/components/features/conversation/conversation-overview-panel.constants";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export function useConversationOverviewLayoutMode(
  containerRef: RefObject<HTMLElement | null>,
  enabled: boolean,
): ConversationOverviewLayoutMode {
  const [layoutMode, setLayoutMode] =
    useState<ConversationOverviewLayoutMode>("hidden");

  useIsomorphicLayoutEffect(() => {
    if (!enabled) {
      setLayoutMode("hidden");
      return undefined;
    }

    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    const update = () => {
      setLayoutMode(getConversationOverviewLayoutMode(container.clientWidth));
    };

    update();

    if (typeof ResizeObserver === "undefined") {
      return undefined;
    }

    const observer = new ResizeObserver(update);
    observer.observe(container);
    return () => observer.disconnect();
  }, [containerRef, enabled]);

  return layoutMode;
}
