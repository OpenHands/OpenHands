import {
  OpenHandsEvent,
  ObservationEvent,
  BaseEvent,
  ExecuteBashAction,
  TerminalAction,
  ExecuteBashObservation,
  PlanningFileEditorObservation,
  TerminalObservation,
  BrowserObservation,
  BrowserNavigateAction,
} from "./core";
import { AgentErrorEvent } from "./core/events/observation-event";
import { MessageEvent } from "./core/events/message-event";
import { ActionEvent } from "./core/events/action-event";
import {
  ConversationStateUpdateEvent,
  ConversationStateUpdateEventAgentStatus,
  ConversationStateUpdateEventFullState,
  ConversationStateUpdateEventStats,
  ConversationErrorEvent,
} from "./core/events/conversation-state-event";
import { SystemPromptEvent } from "./core/events/system-event";
import type { OpenHandsParsedEvent } from "../core/index";

/**
 * Type guard to check if an unknown value is a valid BaseEvent
 * @param value - The value to check
 * @returns true if the value is a valid BaseEvent
 */
export function isBaseEvent(value: unknown): value is BaseEvent {
  return (
    value !== null &&
    typeof value === "object" &&
    "id" in value &&
    "timestamp" in value &&
    "source" in value &&
    typeof value.id === "string" &&
    value.id.length > 0 &&
    typeof value.timestamp === "string" &&
    value.timestamp.length > 0 &&
    typeof value.source === "string" &&
    (value.source === "agent" ||
      value.source === "user" ||
      value.source === "environment")
  );
}

/**
 * Type guard function to check if an event is an observation event
 */
export const isObservationEvent = (
  event: OpenHandsEvent,
): event is ObservationEvent =>
  event.source === "environment" &&
  "action_id" in event &&
  "observation" in event &&
  event.observation !== null &&
  typeof event.observation === "object" &&
  "kind" in event.observation;

/**
 * Type guard function to check if an event is an agent error event
 */
export const isAgentErrorEvent = (
  event: OpenHandsEvent,
): event is AgentErrorEvent =>
  event.source === "agent" &&
  "tool_name" in event &&
  "tool_call_id" in event &&
  "error" in event &&
  typeof event.tool_name === "string" &&
  typeof event.tool_call_id === "string" &&
  typeof event.error === "string";

/**
 * Type guard function to check if an event is a message event (user or assistant)
 */
export const isMessageEvent = (event: OpenHandsEvent): event is MessageEvent =>
  "llm_message" in event &&
  typeof event.llm_message === "object" &&
  event.llm_message !== null &&
  "role" in event.llm_message &&
  "content" in event.llm_message;

/**
 * Type guard function to check if an event is a user message event
 */
export const isUserMessageEvent = (
  event: OpenHandsEvent,
): event is MessageEvent =>
  isMessageEvent(event) && event.llm_message.role === "user";

/**
 * Type guard function to check if an event is an action event
 */
export const isActionEvent = (event: OpenHandsEvent): event is ActionEvent =>
  event.source === "agent" &&
  "action" in event &&
  event.action !== null &&
  typeof event.action === "object" &&
  "kind" in event.action &&
  "tool_name" in event &&
  "tool_call_id" in event &&
  typeof event.tool_name === "string" &&
  typeof event.tool_call_id === "string";

/**
 * Type guard function to check if an action event is an ExecuteBashAction
 */
export const isExecuteBashActionEvent = (
  event: OpenHandsEvent,
): event is ActionEvent<ExecuteBashAction | TerminalAction> =>
  isActionEvent(event) &&
  (event.action.kind === "ExecuteBashAction" ||
    event.action.kind === "TerminalAction");

/**
 * Type guard function to check if an observation event contains terminal output
 */
export const isExecuteBashObservationEvent = (
  event: OpenHandsEvent,
): event is ObservationEvent<ExecuteBashObservation | TerminalObservation> =>
  isObservationEvent(event) &&
  (event.observation.kind === "ExecuteBashObservation" ||
    event.observation.kind === "TerminalObservation");

/**
 * Type guard function to check if an observation event is a PlanningFileEditorObservation
 */
export const isPlanningFileEditorObservationEvent = (
  event: OpenHandsEvent,
): event is ObservationEvent<PlanningFileEditorObservation> =>
  isObservationEvent(event) &&
  event.observation.kind === "PlanningFileEditorObservation";

/**
 * Type guard function to check if an observation event is a BrowserObservation
 */
export const isBrowserObservationEvent = (
  event: OpenHandsEvent,
): event is ObservationEvent<BrowserObservation> =>
  isObservationEvent(event) && event.observation.kind === "BrowserObservation";

/**
 * Type guard function to check if an action event is a BrowserNavigateAction
 */
export const isBrowserNavigateActionEvent = (
  event: OpenHandsEvent,
): event is ActionEvent<BrowserNavigateAction> =>
  isActionEvent(event) && event.action.kind === "BrowserNavigateAction";

/**
 * Type guard function to check if an event is a system prompt event
 */
export const isSystemPromptEvent = (
  event: OpenHandsEvent,
): event is SystemPromptEvent =>
  event.source === "agent" &&
  "system_prompt" in event &&
  "tools" in event &&
  typeof event.system_prompt === "object" &&
  Array.isArray(event.tools);

/**
 * Type guard function to check if an event is a conversation state update event
 */
export const isConversationStateUpdateEvent = (
  event: OpenHandsEvent,
): event is ConversationStateUpdateEvent =>
  "kind" in event && event.kind === "ConversationStateUpdateEvent";

export const isFullStateConversationStateUpdateEvent = (
  event: ConversationStateUpdateEvent,
): event is ConversationStateUpdateEventFullState => event.key === "full_state";

export const isAgentStatusConversationStateUpdateEvent = (
  event: ConversationStateUpdateEvent,
): event is ConversationStateUpdateEventAgentStatus =>
  event.key === "execution_status";

export const isStatsConversationStateUpdateEvent = (
  event: ConversationStateUpdateEvent,
): event is ConversationStateUpdateEventStats => event.key === "stats";

/**
 * Type guard function to check if an event is a conversation error event
 */
export const isConversationErrorEvent = (
  event: OpenHandsEvent,
): event is ConversationErrorEvent =>
  "kind" in event && event.kind === "ConversationErrorEvent";

// =============================================================================
// TEMPORARY COMPATIBILITY TYPE GUARDS
// These will be removed once we fully migrate to V1 events
// =============================================================================

/**
 * TEMPORARY: Type guard to check if an event is a V1 OpenHandsEvent
 * Uses isBaseEvent to validate the complete event structure
 *
 * @deprecated This is temporary until full V1 migration is complete
 */
export function isV1Event(
  event: OpenHandsEvent | OpenHandsParsedEvent,
): event is OpenHandsEvent {
  // Use isBaseEvent to validate the complete BaseEvent structure
  // This ensures the event has all required properties with correct types
  return isBaseEvent(event);
}

/**
 * TEMPORARY: Type guard to check if an event is a V0 OpenHandsParsedEvent
 *
 * @deprecated This is temporary until full V1 migration is complete
 */
export function isV0Event(
  event: OpenHandsEvent | OpenHandsParsedEvent,
): event is OpenHandsParsedEvent {
  // Handle null/undefined cases
  if (!event || typeof event !== "object") {
    return false;
  }

  // V0 events have numeric IDs and either 'action' or 'observation' properties
  return (
    "id" in event &&
    typeof event.id === "number" &&
    ("action" in event || "observation" in event)
  );
}
