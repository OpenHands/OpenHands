import React from "react";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useEventStore } from "#/stores/use-event-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { isV1Event } from "#/types/v1/type-guards";
import { OpenHandsEvent } from "#/types/v1/core";

/**
 * Test component to access and display WebSocket connection state
 */
export function ConnectionStatusComponent() {
  const context = useConversationWebSocket();

  return (
    <div>
      <div data-testid="connection-state">
        {context?.connectionState || "NOT_AVAILABLE"}
      </div>
    </div>
  );
}

/**
 * Test component to access and display event store values
 */
export function EventStoreComponent() {
  const { events, uiEvents } = useEventStore();
  return (
    <div>
      <div data-testid="events-count">{events.length}</div>
      <div data-testid="ui-events-count">{uiEvents.length}</div>
      <div data-testid="latest-event-id">
        {isV1Event(events[events.length - 1])
          ? (events[events.length - 1] as OpenHandsEvent).id
          : "none"}
      </div>
    </div>
  );
}

/**
 * Test component to access and display optimistic user message store values
 */
export function OptimisticUserMessageStoreComponent() {
  const { optimisticUserMessage } = useOptimisticUserMessageStore();
  return (
    <div>
      <div data-testid="optimistic-user-message">
        {optimisticUserMessage || "none"}
      </div>
    </div>
  );
}

/**
 * Test component to access and display error message store values
 */
export function ErrorMessageStoreComponent() {
  const { errorMessage } = useErrorMessageStore();
  return (
    <div>
      <div data-testid="error-message">{errorMessage || "none"}</div>
    </div>
  );
}
