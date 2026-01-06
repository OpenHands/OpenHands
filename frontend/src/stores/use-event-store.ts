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

      const newEvents = [...state.events, event];
      const newEventIds =
        eventId !== undefined
          ? new Set(state.eventIds).add(eventId)
          : state.eventIds;
      const newUiEvents = isV1Event(event)
        ? // @ts-expect-error - temporary, needs proper typing
          handleEventForUI(event, state.uiEvents)
        : [...state.uiEvents, event];

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
