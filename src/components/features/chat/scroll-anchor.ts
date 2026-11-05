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
 */

export interface ScrollAnchor {
  /** The row's `data-row-key`. */
  key: string;
  /** The row's top edge relative to the scroll container's top edge. */
  offset: number;
}

/**
 * The first `[data-row-key]` row whose top is at or below the container's top
 * edge, with its offset from that edge, or `null` when no row is mounted.
 */
export function captureScrollAnchor(element: HTMLElement): ScrollAnchor | null {
  const containerTop = element.getBoundingClientRect().top;
  const rows = Array.from(
    element.querySelectorAll<HTMLElement>("[data-row-key]"),
  );
  for (const row of rows) {
    const key = row.getAttribute("data-row-key");
    if (!key) continue;
    const offset = row.getBoundingClientRect().top - containerTop;
    if (offset >= 0) return { key, offset };
  }
  return null;
}

/**
 * The `scrollTop` that puts the anchored row back at its recorded offset, or
 * `null` when the row is not mounted — the caller then keeps whatever position
 * it already applied.
 */
export function scrollAnchorOffset(
  element: HTMLElement,
  anchor: ScrollAnchor,
): number | null {
  const row = Array.from(
    element.querySelectorAll<HTMLElement>("[data-row-key]"),
  ).find((candidate) => candidate.getAttribute("data-row-key") === anchor.key);
  if (!row) return null;

  const containerTop = element.getBoundingClientRect().top;
  const offset = row.getBoundingClientRect().top - containerTop;
  return element.scrollTop + (offset - anchor.offset);
}
