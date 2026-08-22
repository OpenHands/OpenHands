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
  const eventId = getEventId(event);
  // Transient deltas merge by position and are never persisted/resent, so skip
  // id tracking for them — copying the growing `eventIds` Set per token would
  // otherwise be O(n^2).
  const isDelta = isStreamingDeltaEvent(event);

  // Deduplicate: skip if event with same id already exists (O(1) lookup)
  if (!isDelta && eventId !== undefined && state.eventIds.has(eventId)) {
    return state;
  }

  const newEventIds =
    !isDelta && eventId !== undefined
      ? new Set(state.eventIds).add(eventId)
      : state.eventIds;

  const lastEventIndex = state.events.length - 1;
  const lastEvent = state.events[lastEventIndex];
  const shouldMergeStreamingDelta =
    lastEvent &&
    isDelta &&
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

/**
 * Where a display-ordered insert belongs in `uiEvents`.
 *
 * Scans from the end and lands *after* the last entry stamped no later than
 * `timestamp`, rather than before the first entry stamped later. On a sorted
 * array the two agree; on this one they don't, and only the backward scan is
 * safe: `uiEvents` deliberately ends with a finalized reply above a
 * later-stamped mid-stream message (#1899), and a forward scan would wedge
 * new material into the middle of that pair. Backfilled history still drops
 * into its historical place, because it sorts before both.
 */
const findDisplayInsertIndex = (
  uiEvents: OHEvent[],
  timestamp: string | undefined,
): number => {
  // Undated events go last, matching `compareEventsByTimestamp`.
  if (!timestamp) return uiEvents.length;

  for (let index = uiEvents.length - 1; index >= 0; index -= 1) {
    const other = getEventTimestamp(uiEvents[index]);
    if (other && other.localeCompare(timestamp) <= 0) {
      return index + 1;
    }
  }
  return 0;
};

/**
 * Resorts only `events`, the raw append log: `uiEvents` is display-ordered by
 * handleEventForUI, which deliberately places a finalized reply above a
 * mid-stream message even though the reply's timestamp is later (#1899).
 * Resorting `uiEvents` by raw timestamp here would silently undo that
 * placement the next time any event — e.g. a trailing state snapshot — arrives
 * with a timestamp earlier than the array's current last element.
 */
const sortEvents = (state: EventState): EventState => ({
  ...state,
  events: [...state.events].sort(compareEventsByTimestamp),
});

const applyAddEvent = (state: EventState, event: OHEvent): EventState => {
  const next = appendEvent(state, event);
  if (next === state) {
    return state;
  }

  if (!needsSorting(state.events, event)) {
    return next;
  }

  const sorted = sortEvents(next);

  // `handleEventForUI` appends to the tail. When the event is older than what
  // the store already holds — a reconnect backlog, or the planning
  // sub-conversation replaying its whole history with `resend_mode='all'`
  // after the REST preload seeded newer events — that leaves it rendering at
  // the bottom of the chat while `events` reports it in its real place. Move
  // just that one entry; anything else `handleEventForUI` did (merging a
  // delta, superseding an action, hoisting a finalized reply) is a derivation
  // decision, not an ordering artefact, and must not be second-guessed here.
  const appendedAtTail =
    next.uiEvents.length === state.uiEvents.length + 1 &&
    next.uiEvents[next.uiEvents.length - 1] === event;
  if (!appendedAtTail) {
    return sorted;
  }

  const body = next.uiEvents.slice(0, -1);
  const insertAt = findDisplayInsertIndex(body, getEventTimestamp(event));
  return {
    ...sorted,
    uiEvents: [...body.slice(0, insertAt), event, ...body.slice(insertAt)],
  };
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
      // Derive the incoming page into its *own* display-ordered segment
      // instead of folding it into `state.uiEvents`. `uiEvents` is derived,
      // not raw (see `sortEvents`), and the timestamp sort that used to run
      // here re-ordered the whole array: any older-history page merged by
      // `useLoadOlderEvents` dropped the finalized reply back below the
      // mid-stream message, mid-session and with no reload involved.
      let incomingUiEvents: OHEvent[] = [];
      let earliestAdded: string | undefined;
      let added = false;

      for (const event of incoming) {
        const eventId = getEventId(event);
        // See `appendEvent`: transient deltas are not tracked in `eventIds`.
        const isDelta = isStreamingDeltaEvent(event);
        const isDuplicate =
          !isDelta && eventId !== undefined && eventIds.has(eventId);

        if (!isDuplicate) {
          added = true;
          if (!isDelta && eventId !== undefined) {
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

          const timestamp = getEventTimestamp(event);
          if (
            timestamp &&
            (earliestAdded === undefined ||
              timestamp.localeCompare(earliestAdded) < 0)
          ) {
            earliestAdded = timestamp;
          }

          incomingUiEvents = handleEventForUI(event, incomingUiEvents);
        }
      }

      if (!added) {
        return state;
      }

      // Splice the new segment in as one block. Both sides are already in
      // their own correct display order, and inserting event-by-event would
      // re-sort the segment against itself — undoing any #1899 placement
      // inside it. Both callers (the initial REST load, and older-history
      // pagination, which fetches strictly `timestampLt` the oldest known
      // event) supply a contiguous block that belongs entirely in front of
      // what the store already holds.
      const insertAt = findDisplayInsertIndex(state.uiEvents, earliestAdded);

      return {
        ...sortEvents({ ...state, events, eventIds }),
        uiEvents: [
          ...state.uiEvents.slice(0, insertAt),
          ...incomingUiEvents,
          ...state.uiEvents.slice(insertAt),
        ],
      };
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
