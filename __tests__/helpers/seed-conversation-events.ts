import {
  useEventStore,
  type OHEvent,
} from "#/stores/use-event-store";

/**
 * Seed a conversation's event bucket for tests. Replaces the old flat
 * `useEventStore.setState({ events, uiEvents, eventIds, loadedConversationId })`
 * shape after the store was scoped by conversation id.
 */
export function seedConversationEvents(
  conversationId: string,
  events: OHEvent[],
  uiEvents: OHEvent[] = events,
): void {
  const eventIds = new Set<string | number>();
  for (const event of events) {
    if ("id" in event && event.id !== undefined) {
      eventIds.add(event.id);
    }
  }

  useEventStore.setState((state) => ({
    byConversation: {
      ...state.byConversation,
      [conversationId]: {
        events,
        uiEvents,
        eventIds,
      },
    },
  }));
}

export function clearAllConversationEvents(): void {
  useEventStore.getState().clearEvents();
}
