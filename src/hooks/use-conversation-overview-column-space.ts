import { useEffect, useLayoutEffect, useState, type RefObject } from "react";
import { hasEnoughOverviewLayoutSpace } from "#/components/features/conversation/conversation-overview-panel.constants";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export function useConversationOverviewColumnSpace(
  containerRef: RefObject<HTMLElement | null>,
  enabled: boolean,
): boolean {
  const [hasSpace, setHasSpace] = useState(false);

  useIsomorphicLayoutEffect(() => {
    if (!enabled) {
      setHasSpace(false);
      return undefined;
    }

    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    const update = () => {
      setHasSpace(hasEnoughOverviewLayoutSpace(container.clientWidth));
    };

    update();

    if (typeof ResizeObserver === "undefined") {
      return undefined;
    }

    const observer = new ResizeObserver(update);
    observer.observe(container);
    return () => observer.disconnect();
  }, [containerRef, enabled]);

  return hasSpace;
}
