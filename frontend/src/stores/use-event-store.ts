import { enableMapSet } from "immer";
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { OpenHandsEvent } from "#/types/v1/core";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { OpenHandsParsedEvent } from "#/types/core";
import { isV1Event } from "#/types/v1/type-guards";

// Enable immer support for Set and Map types used in this store
enableMapSet();

// While we transition to v1 events, our store can handle both v0 and v1 events
export type OHEvent = (OpenHandsEvent | OpenHandsParsedEvent) & {
  isFromPlanningAgent?: boolean;
};

const getEventId = (event: OHEvent): string | number | undefined =>
  "id" in event ? event.id : undefined;

const getEventTimestamp = (event: OHEvent): string | undefined =>
  "timestamp" in event ? event.timestamp : undefined;

/**
 * Compare two events by timestamp for sorting.
 * Events without timestamps are placed at the end.
 */
const compareEventsByTimestamp = (a: OHEvent, b: OHEvent): number => {
  const timestampA = getEventTimestamp(a);
  const timestampB = getEventTimestamp(b);

  // Events without timestamps go to the end
  if (!timestampA && !timestampB) return 0;
  if (!timestampA) return 1;
  if (!timestampB) return -1;

  // Compare ISO timestamp strings (lexicographic comparison works for ISO format)
  return timestampA.localeCompare(timestampB);
};

/**
 * Find the correct insertion index using binary search to maintain timestamp order.
 * Returns the index where the event should be inserted.
 */
const findInsertionIndex = (events: OHEvent[], newEvent: OHEvent): number => {
  const newTimestamp = getEventTimestamp(newEvent);
  if (!newTimestamp) return events.length;

  let low = 0;
  let high = events.length;

  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    const midTimestamp = getEventTimestamp(events[mid]);

    if (!midTimestamp || midTimestamp.localeCompare(newTimestamp) <= 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
};

/**
 * Check if the new event is in order (i.e., its timestamp >= the last event's timestamp).
 */
const isInOrder = (events: OHEvent[], newEvent: OHEvent): boolean => {
  if (events.length === 0) return true;

  const lastEvent = events[events.length - 1];
  const lastTimestamp = getEventTimestamp(lastEvent);
  const newTimestamp = getEventTimestamp(newEvent);

  if (!lastTimestamp || !newTimestamp) return true;

  return newTimestamp >= lastTimestamp;
};

/**
 * Insert an event into the array maintaining timestamp order.
 * Uses append when in order (O(1)), binary insertion when out of order (O(log n) search + O(n) shift).
 * Mutates the array in place (safe with immer).
 */
const insertEventOrdered = (events: OHEvent[], event: OHEvent): void => {
  if (isInOrder(events, event)) {
    events.push(event);
  } else {
    const idx = findInsertionIndex(events, event);
    events.splice(idx, 0, event);
  }
};

/**
 * Process a single event for the UI events array.
 * Mutates in place (safe with immer).
 */
const processUiEvent = (event: OHEvent, uiEventsArr: OHEvent[]): OHEvent[] => {
  if (isV1Event(event)) {
    // handleEventForUI returns a new array
    // @ts-expect-error - temporary, needs proper typing
    return handleEventForUI(event, uiEventsArr);
  }
  return [...uiEventsArr, event];
};

export interface EventState {
  events: OHEvent[];
  eventIds: Set<string | number>;
  uiEvents: OHEvent[];
  addEvent: (event: OHEvent) => void;
  addEvents: (events: OHEvent[]) => void;
  clearEvents: () => void;
}

export const useEventStore = create<EventState>()(
  immer((set) => ({
    events: [],
    eventIds: new Set(),
    uiEvents: [],
    addEvent: (event: OHEvent) =>
      set((state) => {
        // Deduplicate: skip if event with same id already exists (O(1) lookup)
        const eventId = getEventId(event);
        if (eventId !== undefined && state.eventIds.has(eventId)) {
          return;
        }

        // Insert event maintaining chronological order
        insertEventOrdered(state.events, event);

        // Track event ID for deduplication
        if (eventId !== undefined) {
          state.eventIds.add(eventId);
        }

        // Process UI events
        state.uiEvents = processUiEvent(event, state.uiEvents);
      }),
    addEvents: (batch: OHEvent[]) =>
      set((state) => {
        if (batch.length === 0) return;

        // Filter duplicates and collect unique events in a single pass
        const uniqueEvents: OHEvent[] = [];
        for (const event of batch) {
          const eventId = getEventId(event);
          if (eventId === undefined || !state.eventIds.has(eventId)) {
            if (eventId !== undefined) {
              state.eventIds.add(eventId);
            }
            uniqueEvents.push(event);
          }
        }

        if (uniqueEvents.length === 0) return;

        // Check if the first new event is in order relative to the existing tail
        if (isInOrder(state.events, uniqueEvents[0])) {
          // Fast path: all batch events are after existing events, just append
          state.events.push(...uniqueEvents);
        } else {
          // Slow path: at least one event is out of order, append and re-sort
          state.events.push(...uniqueEvents);
          state.events.sort(compareEventsByTimestamp);
        }

        // Process UI events for the entire batch
        for (const event of uniqueEvents) {
          state.uiEvents = processUiEvent(event, state.uiEvents);
        }

        // Sort UI events if the batch was out of order
        if (!isInOrder(state.events, uniqueEvents[0])) {
          state.uiEvents.sort(compareEventsByTimestamp);
        }
      }),
    clearEvents: () =>
      set((state) => {
        state.events = [];
        state.eventIds = new Set();
        state.uiEvents = [];
      }),
  })),
);
