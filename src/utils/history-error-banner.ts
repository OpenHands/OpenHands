import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isDisplayableErrorEvent,
  isMessageEvent,
} from "#/types/agent-server/type-guards";

/**
 * Decide whether the conversation's last visible outcome is a failure that the
 * user never saw acknowledged.
 *
 * The REST history preload is the only durable record the UI has after a
 * reload: streaming deltas are transient and the error banner lives in an
 * in-memory store. When the newest event in the loaded tail is a
 * `ConversationErrorEvent` (or `ServerErrorEvent`) with no agent reply or
 * finish after it, the run died unacknowledged and the live-path banner
 * should be re-seeded. Any later agent `MessageEvent` or `FinishAction` means
 * the run recovered (or a later run succeeded) and the banner must stay quiet.
 *
 * Only the loaded history tail is scanned: this matches the data the UI
 * actually has. Older failures outside the window stay quiet.
 */
export const shouldSeedHistoryErrorBanner = (
  events: OpenHandsEvent[],
): boolean => {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    const isAgentReply =
      (isMessageEvent(event) &&
        event.source === "agent" &&
        event.llm_message.role === "assistant") ||
      (isActionEvent(event) && event.action?.kind === "FinishAction");
    if (isAgentReply) {
      return false;
    }
    if (isDisplayableErrorEvent(event)) {
      return true;
    }
  }
  return false;
};
