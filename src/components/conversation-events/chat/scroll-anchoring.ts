import type { VirtualItem } from "@tanstack/react-virtual";
import { BOTTOM_THRESHOLD_PX } from "#/hooks/use-scroll-to-bottom";

export { BOTTOM_THRESHOLD_PX };

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
  const fallback = measuredAverage(heights);
  if (fallback === null) return [];

  // Rows whose height was not captured fall back to the average of the ones
  // that were, so a single unmeasured row cannot distort the total.
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

/**
 * Average of the captured row heights that are usable (positive), or `null`
 * when none were captured. Shared by the seed's fallback and the post-switch
 * estimate so the two agree.
 */
export const measuredAverage = (heights: number[]): number | null => {
  const measured = heights.filter((height) => height > 0);
  if (measured.length === 0) return null;
  return (
    measured.reduce((total, height) => total + height, 0) / measured.length
  );
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

/**
 * Viewport-space top edge of the space a row occupies. A virtual row is a real
 * box, so its own rectangle is the answer. The plain wrapper is
 * `display: contents` and has no box of its own (an empty rectangle), so fall
 * back to its first child's border box minus that child's top margin — the same
 * edge the flex layout gives the row.
 */
export const occupiedRowTop = (wrapper: Element): number => {
  const own = wrapper.getBoundingClientRect();
  if (own.height > 0) return own.top;
  const first = wrapper.firstElementChild;
  if (!first) return own.top;
  return first.getBoundingClientRect().top - verticalMargins(first).top;
};

interface SizeChangeInstance {
  scrollElement: Element | null;
}

/** A row size change to apply once the shell has re-rendered at its new height. */
export interface PendingScrollCorrection {
  /** Signed sum of the (measured - estimated) deltas in this batch. */
  delta: number;
  /** The reader was pinned to the bottom before the change. */
  pinned: boolean;
}

/**
 * Accumulates row size-change corrections between the measuring commit and the
 * shell's re-render at its new total height.
 *
 * The buffer is drained or replaced, never left to accumulate across unrelated
 * changes: a correction a clamp cannot absorb must not be applied to a later
 * size change as a jump.
 */
export class ScrollCorrectionBuffer {
  private pending: PendingScrollCorrection | null = null;

  add(correction: PendingScrollCorrection): void {
    this.pending = this.pending
      ? {
          delta: this.pending.delta + correction.delta,
          pinned: correction.pinned,
        }
      : correction;
  }

  /** The accumulated correction, without clearing it. */
  peek(): PendingScrollCorrection | null {
    return this.pending;
  }

  /** Replaces the accumulated correction; `null` clears it. */
  set(next: PendingScrollCorrection | null): void {
    this.pending = next;
  }

  /** Returns the accumulated correction and clears it. */
  drain(): PendingScrollCorrection | null {
    const pending = this.pending;
    this.pending = null;
    return pending;
  }

  /** Drops any buffered correction (e.g. the reader scrolled manually). */
  clear(): void {
    this.pending = null;
  }
}

/**
 * Whether a row measured at a height other than its estimate needs a scroll
 * correction to keep the reader's place, and whether the reader was pinned to
 * the bottom. A pure decision: the caller accumulates the deltas and applies
 * them once the shell has re-rendered at its new total height.
 *
 * Writing `scrollTop` here instead would clamp: during the commit that measures
 * the row the shell still has the old total height, so `offset + delta` is
 * capped at the old maximum and the excess is lost, with nothing to re-apply
 * it. Several rows correcting in one commit compound the loss.
 *
 * Two cases need a correction: a row above the viewport (`item.start` < the
 * current offset) changing height shifts the content under the viewport, and a
 * row at or below the viewport growing while the reader is pinned to the bottom
 * pushes the bottom away.
 */
export const shouldAnchorScroll = (
  item: Pick<VirtualItem, "start" | "end">,
  delta: number,
  instance: SizeChangeInstance,
): PendingScrollCorrection | null => {
  const element = instance.scrollElement as HTMLElement | null;
  if (!element || delta === 0) return null;

  const offset = element.scrollTop;
  // Pinning is judged from the geometry *before* the change: the shell has not
  // been re-rendered at its new height yet, so `scrollHeight` is still the
  // pre-change total and the new size is not in it.
  const pinned =
    offset + element.clientHeight >= element.scrollHeight - BOTTOM_THRESHOLD_PX;
  if (item.start >= offset && !pinned) return null;

  // Only a row *entirely* above the viewport shifts the content under it. A row
  // that straddles the top edge (its end is below the offset) is what the
  // reader is looking at — a group they just expanded — and the plain list
  // leaves it where it is. Adding its whole growth to `scrollTop` would push
  // the thing they opened out of view.
  if (!pinned && item.end > offset) return null;

  return { delta, pinned };
};

/**
 * Applies a buffered size-change correction to the scroll container and returns
 * the amount consumed from the buffer (so the caller keeps any excess for the
 * next change), or `0` when there is nothing to apply.
 *
 * The correction has to wait until the shell has re-rendered at its new total
 * height: writing it in the measuring commit would clamp `scrollTop` to the old
 * `scrollHeight` and silently lose the excess. `pending.delta` may be negative
 * (rows shrank). When it would push the offset above the new maximum the clamp
 * still applies, and the consumed amount is reported so the caller knows how
 * much actually landed.
 *
 * A pinned reader belongs at the bottom, and the shrink case is why this can't
 * simply add the delta: shrinking a row at the bottom lowers the maximum offset,
 * so the browser clamps `scrollTop` down to it *before* this runs. Adding the
 * (negative) delta again would drop the reader a further `delta` above the
 * newest messages — and off the bottom, turning off auto-follow. Landing on the
 * new maximum absorbs that clamp as the correction and consumes the whole
 * delta, so nothing is retried.
 */
export const applyBufferedScrollCorrection = (
  container: Pick<HTMLElement, "scrollTop" | "scrollHeight" | "clientHeight">,
  pending: PendingScrollCorrection,
): number => {
  if (pending.delta === 0) return 0;
  const element = container;
  const max = Math.max(0, element.scrollHeight - element.clientHeight);
  const target = pending.pinned
    ? max
    : Math.min(Math.max(0, element.scrollTop + pending.delta), max);
  const consumed = pending.pinned ? pending.delta : target - element.scrollTop;
  element.scrollTop = target;
  return consumed;
};
