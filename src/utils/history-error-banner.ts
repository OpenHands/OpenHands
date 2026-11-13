import type { OpenHandsEvent } from "#/types/agent-server/core";
import type {
  ConversationErrorEvent,
  ServerErrorEvent,
} from "#/types/agent-server/core/events/conversation-state-event";
import {
  isActionEvent,
  isConversationErrorEvent,
  isMessageEvent,
  isServerErrorEvent,
} from "#/types/agent-server/type-guards";

/**
 * The error a reloaded conversation should still show in the failure banner:
 * the newest `ConversationErrorEvent`/`ServerErrorEvent` in the loaded history
 * with no agent reply or `FinishAction` after it. The banner store is
 * in-memory and only the live socket writes it, so without this the failure
 * vanishes on reload. Only the loaded tail is scanned.
 */
export const findUnacknowledgedHistoryError = (
  events: readonly OpenHandsEvent[],
): ConversationErrorEvent | ServerErrorEvent | null => {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (isConversationErrorEvent(event) || isServerErrorEvent(event)) {
      return event;
    }
    const isAgentReply =
      (isMessageEvent(event) &&
        event.source === "agent" &&
        event.llm_message.role === "assistant") ||
      (isActionEvent(event) && event.action?.kind === "FinishAction");
    if (isAgentReply) {
      return null;
    }
  }
  return null;
};
