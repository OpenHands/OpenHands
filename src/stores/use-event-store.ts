import { create } from "zustand";
import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  handleEventForUI,
  isSameStreamingSender,
  mergeStreamingDeltaEvent,
} from "#/utils/handle-event-for-ui";
import { isStreamingDeltaEvent } from "#/types/agent-server/type-guards";

export type OHEvent = OpenHandsEvent & {
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
  /**
   * The conversation whose events currently populate the store. The store is
   * global (not keyed by conversation), so the conversation route uses this to
   * tell a genuine conversation switch apart from a remount of the *same*
   * conversation (e.g. navigating to Settings and back) — only the former
   * should clear the accumulated events.
   */
  loadedConversationId: string | null;
  addEvent: (event: OHEvent) => void;
  /**
   * Bulk-insert events. Used for the initial REST history load and for
   * "scroll up to load older" pagination. Newly-added events are de-duped
   * against the existing store and the combined list is re-sorted by
   * timestamp so older pages drop into the correct position.
   */
  addEvents: (events: OHEvent[]) => void;
  /**
   * Clear all events. Also resets `loadedConversationId` to `null` so the
   * store never claims to hold a conversation whose events have been wiped —
   * the invariant (`loadedConversationId` reflects the conversation whose
   * events are in the arrays) holds even for a standalone clear.
   */
  clearEvents: () => void;
  /**
   * Atomically clear all events and record which conversation is now loaded.
   * Collapsing the reset and the bookkeeping into a single `set` keeps the
   * store invariant enforced at the boundary, rather than relying on every
   * call-site to invoke a clear and a `loadedConversationId` setter in the
   * right order.
   */
  clearEventsForConversation: (conversationId: string | null) => void;
}

const appendEvent = (state: EventState, event: OHEvent): EventState => {
  // Deduplicate: skip if event with same id already exists (O(1) lookup)
  const eventId = getEventId(event);
  if (eventId !== undefined && state.eventIds.has(eventId)) {
    return state;
  }

  const newEventIds =
    eventId !== undefined
      ? new Set(state.eventIds).add(eventId)
      : state.eventIds;

  const lastEventIndex = state.events.length - 1;
  const lastEvent = state.events[lastEventIndex];
  const shouldMergeStreamingDelta =
    lastEvent &&
    isStreamingDeltaEvent(event) &&
    isStreamingDeltaEvent(lastEvent) &&
    isSameStreamingSender(event, lastEvent);
  const events = [...state.events];
  if (shouldMergeStreamingDelta) {
    events[lastEventIndex] = mergeStreamingDeltaEvent(event, lastEvent);
  } else {
    events.push(event);
  }

  return {
    ...state,
    events,
    eventIds: newEventIds,
    uiEvents: handleEventForUI(event, state.uiEvents),
  };
};

const sortEventState = (state: EventState): EventState => ({
  ...state,
  events: [...state.events].sort(compareEventsByTimestamp),
  uiEvents: [...state.uiEvents].sort(compareEventsByTimestamp),
});

const applyAddEvent = (state: EventState, event: OHEvent): EventState => {
  const next = appendEvent(state, event);
  if (next === state) {
    return state;
  }

  if (
    !needsSorting(state.events, event) &&
    !needsSorting(state.uiEvents, event)
  ) {
    return next;
  }

  return sortEventState(next);
};

export const useEventStore = create<EventState>()((set) => ({
  events: [],
  eventIds: new Set(),
  uiEvents: [],
  loadedConversationId: null,
  addEvent: (event: OHEvent) => set((state) => applyAddEvent(state, event)),
  addEvents: (incoming: OHEvent[]) =>
    set((state) => {
      if (incoming.length === 0) return state;

      const eventIds = new Set(state.eventIds);
      const events = [...state.events];
      let uiEvents = [...state.uiEvents];
      let added = false;

      for (const event of incoming) {
        const eventId = getEventId(event);
        const isDuplicate = eventId !== undefined && eventIds.has(eventId);

        if (!isDuplicate) {
          added = true;
          if (eventId !== undefined) {
            eventIds.add(eventId);
          }

          const lastEventIndex = events.length - 1;
          const lastEvent = events[lastEventIndex];
          if (
            lastEvent &&
            isStreamingDeltaEvent(event) &&
            isStreamingDeltaEvent(lastEvent) &&
            isSameStreamingSender(event, lastEvent)
          ) {
            events[lastEventIndex] = mergeStreamingDeltaEvent(event, lastEvent);
          } else {
            events.push(event);
          }

          uiEvents = handleEventForUI(event, uiEvents);
        }
      }

      if (!added) {
        return state;
      }

      return sortEventState({
        ...state,
        events,
        eventIds,
        uiEvents,
      });
    }),
  clearEvents: () =>
    set(() => ({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
      loadedConversationId: null,
    })),
  clearEventsForConversation: (conversationId: string | null) =>
    set(() => ({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
      loadedConversationId: conversationId,
    })),
}));

// In dev builds, expose the store on `window` so that fixture/preview
// scripts (e.g. .pr/issue-132 demo capture) can inject synthetic events
// without round-tripping through the agent-server. Tree-shaken in
// production builds via `import.meta.env.DEV`.
if (
  typeof window !== "undefined" &&
  typeof import.meta !== "undefined" &&
  (import.meta as { env?: { DEV?: boolean } }).env?.DEV
) {
  (
    window as unknown as { __OH_EVENT_STORE__?: typeof useEventStore }
  ).__OH_EVENT_STORE__ = useEventStore;
}
