import { OpenHandsParsedEvent } from ".";
import {
  UserMessageAction,
  AssistantMessageAction,
  OpenHandsAction,
  SystemMessageAction,
  CommandAction,
  FinishAction,
  TaskTrackingAction,
} from "./actions";
import {
  AgentStateChangeObservation,
  CommandObservation,
  ErrorObservation,
  MCPObservation,
  OpenHandsObservation,
  TaskTrackingObservation,
} from "./observations";
import { StatusUpdate } from "./variances";

export const isOpenHandsEvent = (
  event: unknown,
): event is OpenHandsParsedEvent =>
  typeof event === "object" &&
  event !== null &&
  "id" in event &&
  "source" in event &&
  "message" in event &&
  "timestamp" in event;

export const isOpenHandsAction = (
  event: OpenHandsParsedEvent,
): event is OpenHandsAction => "action" in event;

export const isOpenHandsObservation = (
  event: OpenHandsParsedEvent,
): event is OpenHandsObservation => "observation" in event;

export const isUserMessage = (
  event: OpenHandsParsedEvent,
): event is UserMessageAction =>
  isOpenHandsAction(event) &&
  event.source === "user" &&
  event.action === "message";

export const isAssistantMessage = (
  event: OpenHandsParsedEvent,
): event is AssistantMessageAction =>
  isOpenHandsAction(event) &&
  event.source === "agent" &&
  (event.action === "message" || event.action === "finish");

export const isErrorObservation = (
  event: OpenHandsParsedEvent,
): event is ErrorObservation =>
  isOpenHandsObservation(event) && event.observation === "error";

export const isCommandAction = (
  event: OpenHandsParsedEvent,
): event is CommandAction => isOpenHandsAction(event) && event.action === "run";

export const isAgentStateChangeObservation = (
  event: OpenHandsParsedEvent,
): event is AgentStateChangeObservation =>
  isOpenHandsObservation(event) && event.observation === "agent_state_changed";

export const isCommandObservation = (
  event: OpenHandsParsedEvent,
): event is CommandObservation =>
  isOpenHandsObservation(event) && event.observation === "run";

export const isFinishAction = (
  event: OpenHandsParsedEvent,
): event is FinishAction =>
  isOpenHandsAction(event) && event.action === "finish";

export const isSystemMessage = (
  event: OpenHandsParsedEvent,
): event is SystemMessageAction =>
  isOpenHandsAction(event) && event.action === "system";

export const isRejectObservation = (
  event: OpenHandsParsedEvent,
): event is OpenHandsObservation =>
  isOpenHandsObservation(event) && event.observation === "user_rejected";

export const isMcpObservation = (
  event: OpenHandsParsedEvent,
): event is MCPObservation =>
  isOpenHandsObservation(event) && event.observation === "mcp";

export const isTaskTrackingAction = (
  event: OpenHandsParsedEvent,
): event is TaskTrackingAction =>
  isOpenHandsAction(event) && event.action === "task_tracking";

export const isTaskTrackingObservation = (
  event: OpenHandsParsedEvent,
): event is TaskTrackingObservation =>
  isOpenHandsObservation(event) && event.observation === "task_tracking";

export const isStatusUpdate = (event: unknown): event is StatusUpdate =>
  typeof event === "object" &&
  event !== null &&
  "status_update" in event &&
  "type" in event &&
  "id" in event;

export const isActionOrObservation = (
  event: OpenHandsParsedEvent,
): event is OpenHandsAction | OpenHandsObservation =>
  isOpenHandsAction(event) || isOpenHandsObservation(event);
