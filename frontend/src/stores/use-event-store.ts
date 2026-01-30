import { create } from "zustand";
import { OpenHandsEvent } from "#/types/v1/core";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { OpenHandsParsedEvent } from "#/types/core";
import { isV1Event } from "#/types/v1/type-guards";

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
 * Check if the new event needs sorting (i.e., it's out of order).
 * Returns true if the new event's timestamp is earlier than the last event's timestamp.
 */
const needsSorting = (events: OHEvent[], newEvent: OHEvent): boolean => {
  if (events.length === 0) return false;

  const lastEvent = events[events.length - 1];
  const lastTimestamp = getEventTimestamp(lastEvent);
  const newTimestamp = getEventTimestamp(newEvent);

  // If either event doesn't have a timestamp, don't sort
  if (!lastTimestamp || !newTimestamp) return false;

  // Sort needed if new event's timestamp is earlier than last event's timestamp
  return newTimestamp < lastTimestamp;
};

export interface EventState {
  events: OHEvent[];
  eventIds: Set<string | number>;
  uiEvents: OHEvent[];
  addEvent: (event: OHEvent) => void;
  clearEvents: () => void;
}

export const useEventStore = create<EventState>()((set) => ({
  events: [],
  eventIds: new Set(),
  uiEvents: [],
  addEvent: (event: OHEvent) =>
    set((state) => {
      // Deduplicate: skip if event with same id already exists (O(1) lookup)
      const eventId = getEventId(event);
      if (eventId !== undefined && state.eventIds.has(eventId)) {
        return state;
      }

      // Add event and sort if needed to maintain chronological order
      let newEvents = [...state.events, event];
      if (needsSorting(state.events, event)) {
        newEvents = newEvents.sort(compareEventsByTimestamp);
      }

      const newEventIds =
        eventId !== undefined
          ? new Set(state.eventIds).add(eventId)
          : state.eventIds;

      // Process UI events and sort if needed
      let newUiEvents = isV1Event(event)
        ? // @ts-expect-error - temporary, needs proper typing
          handleEventForUI(event, state.uiEvents)
        : [...state.uiEvents, event];

      if (needsSorting(state.uiEvents, event)) {
        newUiEvents = newUiEvents.sort(compareEventsByTimestamp);
      }

      return {
        events: newEvents,
        eventIds: newEventIds,
        uiEvents: newUiEvents,
      };
    }),
  clearEvents: () =>
    set(() => ({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
    })),
}));
