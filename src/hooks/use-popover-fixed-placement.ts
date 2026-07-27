import React from "react";

export interface FixedPlacementBox {
  top: number;
  left: number;
  width: number;
}

/**
 * Position a popover with `position: fixed`, anchored below the trigger and
 * clamped within the viewport. Used by the conversation-panel "+ New
 * conversation" menus when they're rendered inside an overflow-hidden
 * sidebar and would otherwise be clipped.
 *
 * Returns the measured `{ top, left, width }` box (or `null` when the
 * popover is closed or fixed placement is disabled). The hook also wires
 * up window resize + capture-phase scroll listeners so the box follows the
 * trigger as the page moves.
 */
export function usePopoverFixedPlacement(
  triggerRef: React.RefObject<HTMLElement | null>,
  options: { open: boolean; enabled: boolean; targetWidth?: number },
): FixedPlacementBox | null {
  const { open, enabled, targetWidth = 16 * 16 } = options;
  const [box, setBox] = React.useState<FixedPlacementBox | null>(null);

  const measure = React.useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const gutter = 8;
    const width = Math.min(targetWidth, window.innerWidth - gutter * 2);
    let left = r.right - width;
    if (left < gutter) left = gutter;
    if (left + width > window.innerWidth - gutter) {
      left = Math.max(gutter, window.innerWidth - gutter - width);
    }
    setBox({ top: r.bottom + 4, left, width });
  }, [triggerRef, targetWidth]);

  React.useLayoutEffect(() => {
    if (!open || !enabled) {
      setBox(null);
      return undefined;
    }
    measure();
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [open, enabled, measure]);

  return box;
}
