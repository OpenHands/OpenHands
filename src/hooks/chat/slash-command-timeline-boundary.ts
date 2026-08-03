import { useEventStore } from "#/stores/use-event-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { toPendingUserMessageBoundary } from "#/stores/slash-command-output-store";
import { matchesPendingConversationId } from "#/utils/pending-task-message-link";

export {
  isPendingUserMessageBoundary,
  toPendingUserMessageBoundary,
} from "#/stores/slash-command-output-store";

/**
 * Capture the latest event in the raw conversation timeline.
 *
 * Slash-command output resolves this immutable boundary against the raw event
 * history when it renders. Unlike `uiEvents`, the raw timeline does not replace
 * an action with its observation or discard a provisional streaming position
 * when the final event arrives.
 */
export const getLastConversationTimelineEventId = (
  conversationId?: string | null,
): string | null => {
  if (conversationId) {
    const pendingMessages =
      useOptimisticUserMessageStore.getState().pendingMessages;
    for (let index = pendingMessages.length - 1; index >= 0; index -= 1) {
      const pending = pendingMessages[index];
      if (
        matchesPendingConversationId(conversationId, pending.conversationId)
      ) {
        return toPendingUserMessageBoundary(pending.id);
      }
    }
  }

  const { events } = useEventStore.getState();

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const eventId = events[index].id;
    if (eventId !== undefined && eventId !== null) return String(eventId);
  }

  return null;
};
