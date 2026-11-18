/**
 * Row-anchored scroll restoration for the chat's "load older" path.
 *
 * The list restores the reader's position from the scroll column's
 * `scrollHeight` delta, which is exact while every row is mounted (the plain
 * list). Once the list virtualizes, the delta is taken in the same commit that
 * prepends the page, when the new top rows are still at the estimate and their
 * real sizes only reach the virtualizer a render later — so the restore is off
 * by the sum of (measured - estimate) over the prepended rows, and the sign
 * varies with the content.
 *
 * Anchoring to a row instead removes that dependency: remember the first row at
 * or below the top edge and its offset, then put it back there once the rows
 * have settled. The second pass is a no-op when the row is not mounted (it
 * virtualized out of the window), so it can only improve the height-delta
 * result.
 *
 * Both list shapes expose the row key on a `[data-row-key]` element — the
 * plain wrapper and the virtual row — and both are measured from the space the
 * row occupies (`occupiedRowTop`), not the element's own rectangle: the plain
 * wrapper is `display: contents` and has no box of its own.
 */

import { occupiedRowTop } from "#/components/conversation-events/chat/scroll-anchoring";

export interface ScrollAnchorCandidate {
  /** The row's `data-row-key`. */
  key: string;
  /** The row's top edge relative to the scroll container's top edge. */
  offset: number;
}

export interface ScrollAnchor {
  /**
   * Rows to re-anchor to, in preference order: the row at the top edge first,
   * then a few rows below it as fallbacks. The top edge row is the one most
   * likely to change key when a page prepends — if it is a tool group and the
   * page boundary falls inside the group's run, the older events merge in and
   * its key moves — so a stable row just below keeps the restore working.
   */
  candidates: ScrollAnchorCandidate[];
}

/** Rows below the top edge to record as fallbacks. */
const ANCHOR_FALLBACK_ROWS = 3;

/**
 * The `[data-row-key]` rows at or below the container's top edge, with their
 * offsets from that edge, in document order. Empty when no row is mounted.
 */
function mountedRowsFromTop(element: HTMLElement): ScrollAnchorCandidate[] {
  const containerTop = element.getBoundingClientRect().top;
  const rows = Array.from(
    element.querySelectorAll<HTMLElement>("[data-row-key]"),
  );
  const candidates: ScrollAnchorCandidate[] = [];
  for (const row of rows) {
    const key = row.getAttribute("data-row-key");
    if (!key) continue;
    const offset = occupiedRowTop(row) - containerTop;
    if (offset >= 0) candidates.push({ key, offset });
    if (candidates.length >= ANCHOR_FALLBACK_ROWS + 1) break;
  }
  return candidates;
}

/**
 * The row at the container's top edge plus a few rows below it, or `null` when
 * no row is mounted.
 */
export function captureScrollAnchor(element: HTMLElement): ScrollAnchor | null {
  const candidates = mountedRowsFromTop(element);
  return candidates.length > 0 ? { candidates } : null;
}

/**
 * The `scrollTop` that puts the first still-mounted candidate back at its
 * recorded offset, or `null` when none of them is mounted — the caller then
 * keeps whatever position it already applied.
 */
export function scrollAnchorOffset(
  element: HTMLElement,
  anchor: ScrollAnchor,
): number | null {
  const byKey = new Map<string, HTMLElement>();
  for (const row of Array.from(
    element.querySelectorAll<HTMLElement>("[data-row-key]"),
  )) {
    const key = row.getAttribute("data-row-key");
    if (key) byKey.set(key, row);
  }

  const containerTop = element.getBoundingClientRect().top;
  for (const candidate of anchor.candidates) {
    const row = byKey.get(candidate.key);
    if (!row) continue;
    const offset = occupiedRowTop(row) - containerTop;
    return element.scrollTop + (offset - candidate.offset);
  }
  return null;
}

/**
 * The `scrollTop` that puts the first candidate the virtualizer knows back at
 * its recorded offset, from the virtualizer's own row starts instead of the
 * DOM, or `null` when it knows none of them.
 *
 * Used when no candidate is mounted: the older page's rows can be taller than
 * the estimate by more than the window, so the anchored row has been pushed
 * out of it and {@link scrollAnchorOffset} cannot find it. The virtualizer
 * still places the row, so scrolling to that place mounts it and the next pass
 * can restore it exactly from the DOM. `rowStart` returns a row's start in the
 * scroll element's coordinates (the same space as `scrollTop`).
 */
export function virtualAnchorOffset(
  anchor: ScrollAnchor,
  rowStart: (key: string) => number | null,
): number | null {
  for (const candidate of anchor.candidates) {
    const start = rowStart(candidate.key);
    if (start === null) continue;
    return Math.max(0, start - candidate.offset);
  }
  return null;
}
