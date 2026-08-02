import { useEventStore } from "#/stores/use-event-store";

/**
 * Capture the latest event in the raw conversation timeline.
 *
 * Slash-command output resolves this immutable boundary against the raw event
 * history when it renders. Unlike `uiEvents`, the raw timeline does not replace
 * an action with its observation or discard a provisional streaming position
 * when the final event arrives.
 */
export const getLastConversationTimelineEventId = (): string | null => {
  const { events } = useEventStore.getState();

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const eventId = events[index].id;
    if (eventId !== undefined && eventId !== null) return String(eventId);
  }

  return null;
};
