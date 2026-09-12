import React from "react";

/**
 * Measures the available viewport space above (or below) the popover trigger
 * and returns a `maxHeight` inline-style value so the menu never grows beyond
 * what's visible on screen.
 *
 * Upward-opening popovers (`position: top` / `bottom-full`) are positioned
 * absolutely relative to the trigger, so when the trigger sits near the bottom
 * of the viewport a `max-h-[60vh]` cap can push the menu's top edge above the
 * screen. The portion above the viewport is clipped by the browser and can't be
 * scrolled back into view (the scroller lives inside the clipped region).
 *
 * Capping `maxHeight` to the measured available space keeps the menu fully
 * visible and lets the user scroll through every row.
 */
export function useAvailablePopoverSpace(
  triggerRef: React.RefObject<HTMLElement | null>,
  options: {
    open: boolean;
    /** Which side of the trigger the popover opens toward. */
    direction: "up" | "down";
    /** Gap between the trigger and the popover, in px. */
    gap?: number;
    /** Upper bound so a tall menu doesn't fill the whole screen. */
    maxHeight?: number;
  },
): number | undefined {
  const { open, direction, gap = 8, maxHeight = 480 } = options;
  const [available, setAvailable] = React.useState<number | undefined>(
    undefined,
  );

  const measure = React.useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const space =
      direction === "up"
        ? Math.max(0, rect.top - gap)
        : Math.max(0, window.innerHeight - rect.bottom - gap);
    setAvailable(Math.min(maxHeight, space));
  }, [triggerRef, direction, gap, maxHeight]);

  React.useLayoutEffect(() => {
    if (!open) {
      setAvailable(undefined);
      return undefined;
    }
    measure();
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [open, measure]);

  return available;
}
