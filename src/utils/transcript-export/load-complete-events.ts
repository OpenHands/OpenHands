import type { OpenHandsEvent } from "#/types/agent-server/core";

export const TRANSCRIPT_HISTORY_PAGE_SIZE = 100;

interface TranscriptEventSearchOptions {
  limit: number;
  sortOrder: "TIMESTAMP_DESC";
  pageId?: string;
  timestampLt?: string;
  strictPagination: true;
}

interface TranscriptEventPage {
  items: OpenHandsEvent[];
  next_page_id?: string | null;
}

type SearchTranscriptEvents = (
  options: TranscriptEventSearchOptions,
) => Promise<TranscriptEventPage>;

const compareEventTimestamps = (
  first: OpenHandsEvent,
  second: OpenHandsEvent,
): number => first.timestamp.localeCompare(second.timestamp);

/**
 * Loads the persisted history from the newest page back to the beginning,
 * then merges any live store events that have not persisted yet. The timestamp
 * anchor matches the chat's existing history pagination, while id-based
 * de-duplication keeps the result stable if pages overlap.
 */
export const loadCompleteTranscriptEvents = async (
  loadedEvents: OpenHandsEvent[],
  searchEvents: SearchTranscriptEvents,
  expectedEventCount?: number,
): Promise<OpenHandsEvent[]> => {
  const persistedDescending: OpenHandsEvent[] = [];
  const fetchedEventIds = new Set<string>();
  const seenPageIds = new Set<string>();
  let oldestTimestamp: string | undefined;
  let pageId: string | undefined;
  let usedCursor = false;
  let usingTimestampFallback = false;

  while (true) {
    const page = await searchEvents({
      limit: TRANSCRIPT_HISTORY_PAGE_SIZE,
      sortOrder: "TIMESTAMP_DESC",
      strictPagination: true,
      ...(pageId ? { pageId } : {}),
      ...(oldestTimestamp ? { timestampLt: oldestTimestamp } : {}),
    });

    if (!Array.isArray(page.items)) {
      throw new Error(
        "Invalid transcript history response: expected page.items to be an array.",
      );
    }

    persistedDescending.push(...page.items);
    let pageOldestTimestamp: string | undefined;
    let addedEvent = false;
    page.items.forEach((event) => {
      if (!fetchedEventIds.has(event.id)) {
        fetchedEventIds.add(event.id);
        addedEvent = true;
      }
      if (!pageOldestTimestamp || event.timestamp < pageOldestTimestamp) {
        pageOldestTimestamp = event.timestamp;
      }
    });

    if (page.next_page_id) {
      if (seenPageIds.has(page.next_page_id)) {
        throw new Error(
          "Transcript history pagination repeated a page cursor.",
        );
      }
      seenPageIds.add(page.next_page_id);
      pageId = page.next_page_id;
      oldestTimestamp = undefined;
      usedCursor = true;
      continue;
    }

    // Once a server supplies a cursor, a page without a next cursor is an
    // explicit exhaustion signal, even when the final page is exactly full.
    if (usedCursor && !usingTimestampFallback) break;
    if (page.items.length < TRANSCRIPT_HISTORY_PAGE_SIZE) break;

    // Some older servers omit cursors for filtered searches. A timestamp
    // fallback is safe to attempt only when an independent event count can
    // prove completeness; otherwise fail instead of exporting a partial tail.
    if (expectedEventCount === undefined) {
      throw new Error(
        "Transcript history pagination cannot prove that all events were loaded.",
      );
    }
    if (!pageOldestTimestamp) {
      throw new Error("Transcript history pagination did not advance.");
    }
    if (
      oldestTimestamp &&
      (!addedEvent || pageOldestTimestamp >= oldestTimestamp)
    ) {
      throw new Error("Transcript history pagination did not advance.");
    }

    pageId = undefined;
    oldestTimestamp = pageOldestTimestamp;
    usingTimestampFallback = true;
  }

  const eventsById = new Map<string, OpenHandsEvent>();
  persistedDescending
    .slice()
    .reverse()
    .forEach((event) => {
      if (!eventsById.has(event.id)) eventsById.set(event.id, event);
    });
  loadedEvents.forEach((event) => {
    if (!eventsById.has(event.id)) eventsById.set(event.id, event);
  });
  // Array.prototype.sort is stable, so equal-timestamp events keep the causal
  // order returned by the server/store rather than being reordered by id.
  const completeEvents = [...eventsById.values()].sort(compareEventTimestamps);
  if (
    expectedEventCount !== undefined &&
    fetchedEventIds.size < expectedEventCount
  ) {
    throw new Error(
      `Transcript history is incomplete: expected ${expectedEventCount} persisted events, received ${fetchedEventIds.size}.`,
    );
  }
  return completeEvents;
};
