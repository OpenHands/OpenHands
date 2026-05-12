import { OpenHandsParsedEvent } from "#/types/core";
import { OpenHandsEvent } from "#/types/v1/core";
import {
  isActionEvent,
  isObservationEvent,
  isMessageEvent,
  isAgentErrorEvent,
  isConversationStateUpdateEvent,
  isHookExecutionEvent,
  isACPToolCallEvent,
  isV1Event,
} from "#/types/v1/type-guards";

export const shouldRenderEvent = (event: OpenHandsEvent) => {
  // Explicitly exclude system events that should not be rendered in chat
  if (isConversationStateUpdateEvent(event)) {
    return false;
  }

  // Render action events (with filtering)
  if (isActionEvent(event)) {
    // For V1, action is an object with kind property
    const actionType = event.action.kind;

    if (!actionType) {
      return false;
    }

    // Hide user commands from the chat interface
    if (actionType === "ExecuteBashAction" && event.source === "user") {
      return false;
    }

    // Hide PlanningFileEditorAction - handled separately with PlanPreview component
    if (actionType === "PlanningFileEditorAction") {
      return false;
    }

    return true;
  }

  // Render observation events
  if (isObservationEvent(event)) {
    return true;
  }

  // Render message events (user and assistant messages)
  if (isMessageEvent(event)) {
    return true;
  }

  // Render agent error events
  if (isAgentErrorEvent(event)) {
    return true;
  }

  // Render hook execution events
  if (isHookExecutionEvent(event)) {
    return true;
  }

  // Render ACP sub-agent tool call events
  if (isACPToolCallEvent(event)) {
    return true;
  }

  // Don't render any other event types (system events, etc.)
  return false;
};

export const hasUserEvent = (events: OpenHandsEvent[]) =>
  events.some((event) => event.source === "user");

/**
 * Narrow a mixed V0/V1 event list to V1 events that actually render in chat.
 * Single source of truth: callers (e.g. `useFilteredEvents`, slash-command
 * interceptors that anchor to the latest visible event) MUST use this rather
 * than re-implementing `isV1Event` + `shouldRenderEvent` chains, so updates
 * to the rendering rules are picked up everywhere.
 */
export const getRenderedV1Events = (
  events: ReadonlyArray<OpenHandsEvent | OpenHandsParsedEvent>,
): OpenHandsEvent[] => events.filter(isV1Event).filter(shouldRenderEvent);
