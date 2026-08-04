import { detectAgentNotificationsFromEvents } from "#/components/features/chat/detect-agent-notifications-from-events";
import { extractAgentNotifications } from "#/components/features/chat/agent-notifications-parser";
import { parseMessageFromEvent } from "#/components/conversation-events/chat/event-content-helpers/parse-message-from-event";
import type { AgentNotification } from "#/components/features/chat/agent-notifications.constants";
import type { OHEvent } from "#/stores/use-event-store";
import { isMessageEvent } from "#/types/agent-server/type-guards";

/**
 * Collects skill/automation recommendations from the current conversation
 * event list: assistant-message fences plus tool-use heuristics.
 */
export function collectAgentNotificationsFromEvents(
  events: OHEvent[],
): AgentNotification[] {
  const byId = new Map<string, AgentNotification>();

  for (const event of events) {
    if (!isMessageEvent(event) || event.source !== "agent") {
      continue;
    }

    const { notifications } = extractAgentNotifications(
      parseMessageFromEvent(event),
    );
    for (const notification of notifications) {
      byId.set(notification.id, notification);
    }
  }

  for (const notification of detectAgentNotificationsFromEvents(events)) {
    if (!byId.has(notification.id)) {
      byId.set(notification.id, notification);
    }
  }

  return [...byId.values()];
}
