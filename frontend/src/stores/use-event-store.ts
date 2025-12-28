import { create } from "zustand";
import { OpenHandsEvent } from "#/types/v1/core";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { OpenHandsParsedEvent } from "#/types/core";
import { isV1Event } from "#/types/v1/type-guards";

// While we transition to v1 events, our store can handle both v0 and v1 events
type OHEvent = (OpenHandsEvent | OpenHandsParsedEvent) & {
  isFromPlanningAgent?: boolean;
};

interface EventState {
  events: OHEvent[];
  uiEvents: OHEvent[];
  addEvent: (event: OHEvent) => void;
  clearEvents: () => void;
}

export const useEventStore = create<EventState>()((set) => ({
  events: [],
  uiEvents: [],
  addEvent: (event: OHEvent) =>
    set((state) => {
      const newEvents = [...state.events, event];
      const newUiEvents = isV1Event(event)
        ? // @ts-expect-error - temporary, needs proper typing
          handleEventForUI(event, state.uiEvents)
        : [...state.uiEvents, event];

      return {
        events: newEvents,
        uiEvents: newUiEvents,
      };
    }),
  clearEvents: () =>
    set(() => ({
      events: [],
      uiEvents: [],
    })),
}));
