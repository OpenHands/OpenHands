import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import { useEventStore } from "#/stores/use-event-store";

export const getLastRenderableEventId = (): string | null => {
  const { uiEvents } = useEventStore.getState();

  for (let index = uiEvents.length - 1; index >= 0; index -= 1) {
    const event = uiEvents[index];
    if (shouldRenderEvent(event)) return String(event.id);
  }

  return null;
};
