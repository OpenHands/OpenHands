import React from "react";
import { useEventStore } from "#/stores/use-event-store";
import {
  shouldRenderEvent as shouldRenderAgentServerEvent,
  hasUserEvent as hasAgentServerUserEvent,
} from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import {
  isSystemPromptEvent,
  isConversationStateUpdateEvent,
} from "#/types/agent-server/type-guards";

/**
 * Hook that provides memoized filtered event arrays for ChatInterface.
 */
export function useFilteredEvents() {
  const storeEvents = useEventStore((state) => state.events);
  const uiEvents = useEventStore((state) => state.uiEvents);

  const renderableEvents = React.useMemo(
    () => uiEvents.filter(shouldRenderAgentServerEvent),
    [uiEvents],
  );

  const allConversationEvents = React.useMemo(() => storeEvents, [storeEvents]);

  const totalEvents = React.useMemo(
    () => renderableEvents.length,
    [renderableEvents],
  );

  const hasSubstantiveAgentActions = React.useMemo(
    () =>
      allConversationEvents.some(
        (event) =>
          event.source === "agent" &&
          !isSystemPromptEvent(event) &&
          !isConversationStateUpdateEvent(event),
      ),
    [allConversationEvents],
  );

  const conversationUserEventsExist = hasAgentServerUserEvent(
    allConversationEvents,
  );

  return {
    storeEvents,
    uiEvents,
    renderableEvents,
    allConversationEvents,
    totalEvents,
    hasSubstantiveAgentActions,
    conversationUserEventsExist,
    userEventsExist: conversationUserEventsExist,
  };
}
