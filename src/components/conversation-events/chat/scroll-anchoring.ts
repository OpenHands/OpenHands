import type { VirtualItem } from "@tanstack/react-virtual";

/** Distance from the bottom (px) still treated as "pinned to the bottom". */
export const BOTTOM_THRESHOLD_PX = 20;

/**
 * Seeds the virtualizer's measurement cache from the plain list's own row
 * heights.
 *
 * Crossing the virtualization threshold replaces the plain list (real heights)
 * with the shell (one flat estimate per row). If the estimate's total is
 * shorter than the plain list, the browser clamps `scrollTop` to the new
 * maximum and yanks a reader who was part-way up the history toward the bottom.
 * Seeding every row at the height it just had keeps the shell's total height —
 * and so the scroll offset — unchanged across the switch, and the live
 * measurements then correct any drift in place.
 *
 * Returns `[]` when no heights were captured (or none are usable), which leaves
 * the virtualizer on its own estimate.
 */
export const buildInitialMeasurements = (
  keys: string[],
  heights: number[],
  gapPx: number,
): VirtualItem[] => {
  if (keys.length === 0) return [];
  if (!heights.some((height) => height > 0)) return [];

  // Rows whose height was not captured fall back to the average of the ones
  // that were, so a single unmeasured row cannot distort the total.
  const measured = heights.filter((height) => height > 0);
  const fallback = measured.length
    ? measured.reduce((total, height) => total + height, 0) / measured.length
    : 0;

  let start = 0;
  return keys.map((key, index) => {
    const size = Math.max(1, Math.round(heights[index] || fallback)) + gapPx;
    const item: VirtualItem = {
      index,
      key,
      size,
      lane: 0,
      start,
      end: start + size,
    };
    start += size;
    return item;
  });
};

/** Vertical margins of an element, in px (0 when there is no CSSOM). */
const verticalMargins = (element: Element): { top: number; bottom: number } => {
  if (typeof getComputedStyle !== "function") return { top: 0, bottom: 0 };
  const style = getComputedStyle(element);
  return {
    top: Number.parseFloat(style.marginTop) || 0,
    bottom: Number.parseFloat(style.marginBottom) || 0,
  };
};

/**
 * Height a plain row occupies in the scroll column, in the same terms the
 * virtualizer's `measureElement` reports for a virtual row.
 *
 * The plain row's wrapper is `display: contents`, so it has no box of its own
 * and the row's extent is the span of its children. A bounding rectangle is the
 * *border* box, which leaves out those children's vertical margins — a user
 * bubble carries `mt-6`, the last child carries `mb-4` — while the virtual row's
 * box contains them. Seeding the shell from the rectangles alone therefore
 * under-states every row with a margin, the shell comes up shorter than the
 * plain list it replaced, and the browser clamps `scrollTop` toward the bottom,
 * moving a reader who was part-way up the history. Include the margins so the
 * seed matches what the virtual rows report.
 */
const occupiedRowHeight = (first: Element, last: Element): number => {
  const top = first.getBoundingClientRect().top - verticalMargins(first).top;
  const bottom =
    last.getBoundingClientRect().bottom + verticalMargins(last).bottom;
  return bottom - top;
};

/**
 * Measures one plain row (a `data-row-key` wrapper whose children are the row's
 * boxes) and returns its key with the height it occupies. Returns `null` for a
 * wrapper that has no key or no children yet.
 */
export const measurePlainRow = (wrapper: Element): [string, number] | null => {
  const rowKey = (wrapper as HTMLElement).dataset.rowKey;
  const first = wrapper.firstElementChild;
  const last = wrapper.lastElementChild;
  if (!rowKey || !first || !last) return null;
  return [rowKey, occupiedRowHeight(first, last)];
};

interface SizeChangeInstance {
  scrollElement: Element | null;
}

/**
 * Restores the reader's place when a row is measured at a height other than its
 * estimate, then reports `false` so the virtualizer does not add its own
 * correction on top.
 *
 * Two cases need a correction: a row above the viewport (`item.start` < the
 * current offset) changing height shifts the content under the viewport, and a
 * row at or below the viewport growing while the reader is pinned to the bottom
 * pushes the bottom away.
 *
 * TanStack's default decides both from `getScrollOffset()` — the position cached
 * from the last `scroll` event — and applies the delta to that same cached
 * position. The chat also writes `scrollTop` directly (stick-to-bottom), in the
 * same frame as these corrections, so the cache is stale by exactly that jump
 * and the correction drags the list back to it: a pinned reader lands short of
 * the bottom and autoscroll is then switched off.
 *
 * Reading the element's live position and adding the row's own `delta` once
 * keeps every correction relative to where the reader actually is, and several
 * rows correcting in one frame compose because each reads the position the
 * previous one wrote.
 */
export const anchorScrollOnItemSizeChange = (
  item: Pick<VirtualItem, "start">,
  delta: number,
  instance: SizeChangeInstance,
): boolean => {
  const element = instance.scrollElement as HTMLElement | null;
  if (!element || delta === 0) return false;

  const offset = element.scrollTop;
  // Pinning is judged from the geometry *before* the change: the shell has not
  // been re-rendered at its new height yet, so `scrollHeight` is still the
  // pre-change total and the new size is not in it.
  const pinned =
    offset + element.clientHeight >= element.scrollHeight - BOTTOM_THRESHOLD_PX;
  if (item.start >= offset && !pinned) return false;

  element.scrollTop = offset + delta;
  return false;
};
