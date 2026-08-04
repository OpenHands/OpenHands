import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import { getConversationUiEvents } from "#/stores/use-event-store";

export const getLastRenderableEventId = (
  conversationId: string | null | undefined,
): string | null => {
  if (!conversationId) return null;

  const uiEvents = getConversationUiEvents(conversationId);

  for (let index = uiEvents.length - 1; index >= 0; index -= 1) {
    const event = uiEvents[index];
    if (shouldRenderEvent(event)) return String(event.id);
  }

  return null;
};
