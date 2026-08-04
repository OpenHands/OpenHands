import { useEffect, useRef } from "react";
import { collectAgentNotificationsFromEvents } from "#/components/features/chat/collect-agent-notifications-from-events";
import { extractAgentNotifications } from "#/components/features/chat/agent-notifications-parser";
import { parseMessageFromEvent } from "#/components/conversation-events/chat/event-content-helpers/parse-message-from-event";
import { useAgentState } from "#/hooks/use-agent-state";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";
import { AgentState } from "#/types/agent-state";
import {
  isMessageEvent,
  isSystemPromptEvent,
  isConversationStateUpdateEvent,
} from "#/types/agent-server/type-guards";

function getEventId(event: OHEvent): string | number | undefined {
  return "id" in event ? event.id : undefined;
}

function hasSubstantiveAgentActions(events: OHEvent[]): boolean {
  return events.some(
    (event) =>
      event.source === "agent" &&
      !isSystemPromptEvent(event) &&
      !isConversationStateUpdateEvent(event),
  );
}

/**
 * Ingests agent-notification recommendations from the live conversation:
 * - fenced blocks in new assistant messages (detection-skill contract)
 * - heuristic detection when the agent returns to AWAITING_USER_INPUT
 */
export function useIngestAgentNotificationsFromEvents(
  conversationId: string | null,
) {
  const events = useEventStore((state) => state.events);
  const { curAgentState } = useAgentState();
  const ensureHydrated = useAgentNotificationsStore(
    (state) => state.ensureHydrated,
  );
  const addNotifications = useAgentNotificationsStore(
    (state) => state.addNotifications,
  );

  const processedMessageEventIdsRef = useRef<Set<string | number>>(new Set());
  const lastHeuristicSignatureRef = useRef<string | null>(null);

  useEffect(() => {
    if (conversationId) {
      ensureHydrated(conversationId);
    }
  }, [conversationId, ensureHydrated]);

  useEffect(() => {
    processedMessageEventIdsRef.current = new Set();
    lastHeuristicSignatureRef.current = null;
  }, [conversationId]);

  useEffect(() => {
    if (!conversationId) {
      return;
    }

    for (const event of events) {
      if (!isMessageEvent(event) || event.source !== "agent") {
        continue;
      }

      const eventId = getEventId(event);
      if (
        eventId !== undefined &&
        processedMessageEventIdsRef.current.has(eventId)
      ) {
        continue;
      }

      const text = parseMessageFromEvent(event);
      const { notifications } = extractAgentNotifications(text);
      if (notifications.length > 0) {
        addNotifications(conversationId, notifications);
      }

      if (eventId !== undefined) {
        processedMessageEventIdsRef.current.add(eventId);
      }
    }
  }, [addNotifications, conversationId, events]);

  useEffect(() => {
    if (!conversationId) {
      return;
    }

    if (curAgentState !== AgentState.AWAITING_USER_INPUT) {
      return;
    }

    if (!hasSubstantiveAgentActions(events)) {
      return;
    }

    const signature = events
      .map((event) => String(getEventId(event) ?? event.timestamp ?? ""))
      .join("|");
    if (signature === lastHeuristicSignatureRef.current) {
      return;
    }

    const detected = collectAgentNotificationsFromEvents(events);
    if (detected.length > 0) {
      addNotifications(conversationId, detected);
    }

    lastHeuristicSignatureRef.current = signature;
  }, [addNotifications, conversationId, curAgentState, events]);
}
