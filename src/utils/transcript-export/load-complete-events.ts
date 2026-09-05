import type { OpenHandsEvent } from "#/types/agent-server/core";
import type { TranscriptTruncation } from "./index";

export const TRANSCRIPT_HISTORY_PAGE_SIZE = 100;

/**
 * Above this many events, the export loads a bounded head+tail window instead
 * of the full history, so a 60k-event conversation never materializes in
 * memory. Kept intentionally generous: enyst tested 7k-event exports fine.
 */
export const MAX_TRANSCRIPT_EXPORT_EVENTS = 10_000;
/** Oldest events kept when a large conversation is truncated (task setup). */
export const TRANSCRIPT_HEAD_EVENTS = 2_000;
/** Newest events kept when a large conversation is truncated (recent context). */
export const TRANSCRIPT_TAIL_EVENTS =
  MAX_TRANSCRIPT_EXPORT_EVENTS - TRANSCRIPT_HEAD_EVENTS;

interface TranscriptEventSearchOptions {
  limit: number;
  sortOrder: "TIMESTAMP" | "TIMESTAMP_DESC";
  pageId?: string;
  timestampLt?: string;
  timestampGte?: string;
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
): number => (first.timestamp ?? "").localeCompare(second.timestamp ?? "");

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
  // Id-less events can't be de-duplicated by id, so keep a separate ordered
  // list and a count so they survive the merge and count toward completeness.
  const idLessEvents: OpenHandsEvent[] = [];
  let idLessPersisted = 0;
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
      if (!fetchedEventIds.has(event.id ?? "")) {
        if (event.id) {
          fetchedEventIds.add(event.id);
        } else {
          idLessPersisted += 1;
        }
        addedEvent = true;
      }
      if (
        !pageOldestTimestamp ||
        (event.timestamp ?? "") < pageOldestTimestamp
      ) {
        pageOldestTimestamp = event.timestamp ?? "";
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
      if (event.id !== undefined) {
        if (!eventsById.has(event.id)) {
          eventsById.set(event.id, event);
        }
        return;
      }
      idLessEvents.push(event);
    });
  loadedEvents.forEach((event) => {
    if (event.id !== undefined) {
      if (!eventsById.has(event.id)) {
        eventsById.set(event.id, event);
      }
      return;
    }
    idLessEvents.push(event);
  });
  // Array.prototype.sort is stable, so equal-timestamp events keep the causal
  // order returned by the server/store rather than being reordered by id.
  const completeEvents = [...idLessEvents, ...eventsById.values()].sort(
    compareEventTimestamps,
  );
  if (
    expectedEventCount !== undefined &&
    fetchedEventIds.size + idLessPersisted < expectedEventCount
  ) {
    throw new Error(
      `Transcript history is incomplete: expected ${expectedEventCount} persisted events, received ${fetchedEventIds.size + idLessPersisted}.`,
    );
  }
  return completeEvents;
};

/**
 * Pages in a single direction until `maxEvents` are collected or the server
 * signals exhaustion. Mirrors the cursor / timestamp anchoring in
 * loadCompleteTranscriptEvents but stops at a hard cap, so an enormous history
 * is never loaded in full. Descending order walks newest->oldest (the tail);
 * ascending order walks oldest->newest (the head).
 */
const fetchBoundedTranscriptEvents = async (
  searchEvents: SearchTranscriptEvents,
  sortOrder: "TIMESTAMP" | "TIMESTAMP_DESC",
  maxEvents: number,
): Promise<OpenHandsEvent[]> => {
  const collected: OpenHandsEvent[] = [];
  const fetchedEventIds = new Set<string>();
  const seenPageIds = new Set<string>();
  const descending = sortOrder === "TIMESTAMP_DESC";
  let boundaryTimestamp: string | undefined;
  let pageId: string | undefined;

  while (collected.length < maxEvents) {
    const page = await searchEvents({
      limit: Math.min(
        TRANSCRIPT_HISTORY_PAGE_SIZE,
        maxEvents - collected.length,
      ),
      sortOrder,
      strictPagination: true,
      ...(pageId ? { pageId } : {}),
      ...(boundaryTimestamp
        ? descending
          ? { timestampLt: boundaryTimestamp }
          : { timestampGte: boundaryTimestamp }
        : {}),
    });

    if (!Array.isArray(page.items)) {
      throw new Error(
        "Invalid transcript history response: expected page.items to be an array.",
      );
    }

    let addedEvent = false;
    let pageBoundaryTimestamp: string | undefined;
    for (const event of page.items) {
      if (event.id === undefined || !fetchedEventIds.has(event.id)) {
        if (event.id !== undefined) fetchedEventIds.add(event.id);
        collected.push(event);
        addedEvent = true;
      }
      if (
        !pageBoundaryTimestamp ||
        (descending
          ? (event.timestamp ?? "") < pageBoundaryTimestamp
          : (event.timestamp ?? "") > pageBoundaryTimestamp)
      ) {
        pageBoundaryTimestamp = event.timestamp;
      }
      if (collected.length >= maxEvents) break;
    }
    if (collected.length >= maxEvents) break;

    if (page.next_page_id) {
      if (seenPageIds.has(page.next_page_id)) {
        throw new Error(
          "Transcript history pagination repeated a page cursor.",
        );
      }
      seenPageIds.add(page.next_page_id);
      pageId = page.next_page_id;
      boundaryTimestamp = undefined;
      continue;
    }

    // A short page means the server has no more events in this direction.
    if (page.items.length < TRANSCRIPT_HISTORY_PAGE_SIZE) break;
    // Full page without a cursor: fall back to a timestamp anchor, but bail if
    // it cannot advance (e.g. a whole page sharing one timestamp) so the loop
    // is always bounded.
    if (!addedEvent || !pageBoundaryTimestamp) break;
    pageId = undefined;
    boundaryTimestamp = pageBoundaryTimestamp;
  }

  return collected;
};

/**
 * Loads a bounded head+tail window for very large conversations: the oldest
 * TRANSCRIPT_HEAD_EVENTS and newest TRANSCRIPT_TAIL_EVENTS, plus any live store
 * events, without ever paging the omitted middle. Returns the merged, ordered
 * events and a `truncation` descriptor the renderer uses to place the omission
 * notice. When nothing was actually omitted, `truncation` is left undefined.
 */
export const loadBoundedTranscriptEvents = async (
  loadedEvents: OpenHandsEvent[],
  searchEvents: SearchTranscriptEvents,
  totalEventCount: number,
  headMax: number = TRANSCRIPT_HEAD_EVENTS,
  tailMax: number = TRANSCRIPT_TAIL_EVENTS,
): Promise<{ events: OpenHandsEvent[]; truncation?: TranscriptTruncation }> => {
  const tail = await fetchBoundedTranscriptEvents(
    searchEvents,
    "TIMESTAMP_DESC",
    tailMax,
  );
  const head = await fetchBoundedTranscriptEvents(
    searchEvents,
    "TIMESTAMP",
    headMax,
  );

  const headIds = new Set(head.map((event) => event.id ?? event));
  const eventsById = new Map<string | OpenHandsEvent, OpenHandsEvent>();
  for (const event of head) eventsById.set(event.id ?? event, event);
  for (const event of tail) {
    if (!eventsById.has(event.id ?? event))
      eventsById.set(event.id ?? event, event);
  }
  // Live store events are the newest; keep any not already fetched.
  for (const event of loadedEvents) {
    if (!eventsById.has(event.id ?? event))
      eventsById.set(event.id ?? event, event);
  }

  const events = [...eventsById.values()].sort(compareEventTimestamps);

  // Head events (oldest) sort first; count the leading run so the renderer
  // knows where the head ends and the omission notice belongs.
  let headEventCount = 0;
  for (const event of events) {
    if (!headIds.has(event.id ?? event)) break;
    headEventCount += 1;
  }

  const omittedCount = Math.max(0, totalEventCount - eventsById.size);
  if (omittedCount === 0) {
    return { events };
  }
  return { events, truncation: { omittedCount, headEventCount } };
};
