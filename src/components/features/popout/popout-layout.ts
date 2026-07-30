import {
  POPOUT_DOCK_INSET_PX,
  POPOUT_EXPANDED_WIDTH_PX,
  POPOUT_GAP_PX,
  POPOUT_MINIMIZED_WIDTH_PX,
  POPOUT_OVERFLOW_BUTTON_SIZE_PX,
  type Popout,
} from "#/stores/popout-store";

export interface PopoutLayout {
  /** Popouts that fit in the dock, ordered oldest-to-newest from left to right. */
  visible: Popout[];
  /** Overflowed popouts, ordered most-recently-used first for the selector. */
  hidden: Popout[];
}

/**
 * Keep active, recently used popouts visible without changing their explicit
 * expanded/minimized state. Explicitly minimized entries rank below expanded
 * ones; recency breaks ties. Once any popout overflows, reserve room for a
 * fixed-size selector at the far right and move lower-priority entries into
 * it. Selecting a hidden popout expands and refreshes it, causing it to
 * displace the lowest-priority visible entry on the next layout pass.
 */
export function layoutPopouts(
  popouts: Popout[],
  viewportWidth: number,
): PopoutLayout {
  if (popouts.length === 0) return { visible: [], hidden: [] };

  const available = Math.max(0, viewportWidth - POPOUT_DOCK_INSET_PX * 2);
  const widthOf = (entry: Popout) =>
    entry.mode === "expanded"
      ? POPOUT_EXPANDED_WIDTH_PX
      : POPOUT_MINIMIZED_WIDTH_PX;
  const widthFor = (entries: Popout[]) =>
    entries.reduce(
      (total, entry, index) =>
        total + widthOf(entry) + (index > 0 ? POPOUT_GAP_PX : 0),
      0,
    );
  const oldestFirst = [...popouts].sort((a, b) => a.openedAt - b.openedAt);

  if (widthFor(popouts) <= available) {
    return { visible: oldestFirst, hidden: [] };
  }

  const availableForPopouts = Math.max(
    0,
    available - POPOUT_OVERFLOW_BUTTON_SIZE_PX - POPOUT_GAP_PX,
  );
  const mostActiveFirst = [...popouts].sort((a, b) => {
    if (a.mode !== b.mode) return a.mode === "expanded" ? -1 : 1;
    return b.openedAt - a.openedAt;
  });
  const visibleIds = new Set<string>();
  let usedWidth = 0;

  for (const entry of mostActiveFirst) {
    const entryWidth =
      widthOf(entry) + (visibleIds.size > 0 ? POPOUT_GAP_PX : 0);
    if (usedWidth + entryWidth <= availableForPopouts) {
      visibleIds.add(entry.conversationId);
      usedWidth += entryWidth;
    } else {
      break;
    }
  }

  // On unusually narrow viewports, retaining the newest entry is still more
  // useful than showing only a count. CSS constrains its width in the host.
  if (visibleIds.size === 0) {
    const mostActive = mostActiveFirst[0];
    if (mostActive) visibleIds.add(mostActive.conversationId);
  }

  return {
    visible: oldestFirst.filter((entry) =>
      visibleIds.has(entry.conversationId),
    ),
    hidden: mostActiveFirst.filter(
      (entry) => !visibleIds.has(entry.conversationId),
    ),
  };
}
