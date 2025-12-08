import { V0_WebSocketStatus } from "#/context/ws-client-provider";
import { I18nKey } from "#/i18n/declaration";
import { AgentState } from "#/types/agent-state";
import { ConversationStatus } from "#/types/conversation-status";
import { StatusMessage } from "#/types/message";
import { RuntimeStatus } from "#/types/runtime-status";
import { V1AppConversationStartTaskStatus } from "#/api/conversation-service/v1-conversation-service.types";
import { isTaskPolling } from "./utils";

export enum IndicatorColor {
  BLUE = "bg-blue-500",
  GREEN = "bg-green-500",
  ORANGE = "bg-orange-500",
  YELLOW = "bg-yellow-500",
  RED = "bg-red-500",
  DARK_ORANGE = "bg-orange-800",
}

export const AGENT_STATUS_MAP: {
  [k: string]: string;
} = {
  // Initializing states
  [AgentState.LOADING]: I18nKey.AGENT_STATUS$INITIALIZING,
  [AgentState.INIT]: I18nKey.AGENT_STATUS$INITIALIZING,

  // Ready/Idle/Waiting for user input states
  [AgentState.AWAITING_USER_INPUT]: I18nKey.AGENT_STATUS$WAITING_FOR_TASK,
  [AgentState.AWAITING_USER_CONFIRMATION]:
    I18nKey.AGENT_STATUS$WAITING_FOR_USER_CONFIRMATION,
  [AgentState.USER_CONFIRMED]: I18nKey.AGENT_STATUS$WAITING_FOR_TASK,
  [AgentState.USER_REJECTED]: I18nKey.AGENT_STATUS$WAITING_FOR_TASK,
  [AgentState.FINISHED]: I18nKey.AGENT_STATUS$WAITING_FOR_TASK,

  // Actively working states
  [AgentState.RUNNING]: I18nKey.AGENT_STATUS$RUNNING_TASK,

  // Agent stopped/paused states
  [AgentState.PAUSED]: I18nKey.AGENT_STATUS$AGENT_STOPPED,
  [AgentState.STOPPED]: I18nKey.AGENT_STATUS$AGENT_STOPPED,
  [AgentState.REJECTED]: I18nKey.AGENT_STATUS$AGENT_STOPPED,

  // Agent error states
  [AgentState.ERROR]: I18nKey.AGENT_STATUS$ERROR_OCCURRED,
  [AgentState.RATE_LIMITED]: I18nKey.AGENT_STATUS$ERROR_OCCURRED,
};

export function getIndicatorColor(
  webSocketStatus: V0_WebSocketStatus,
  conversationStatus: ConversationStatus | null,
  runtimeStatus: RuntimeStatus | null,
  agentState: AgentState | null,
) {
  if (
    webSocketStatus === "DISCONNECTED" ||
    conversationStatus === "STOPPED" ||
    runtimeStatus === "STATUS$STOPPED" ||
    agentState === AgentState.STOPPED ||
    agentState === AgentState.ERROR
  ) {
    return IndicatorColor.RED;
  }

  // Prioritize agent state when it indicates readiness, even if runtime status is stale
  const agentIsReady =
    agentState &&
    [
      AgentState.AWAITING_USER_INPUT,
      AgentState.RUNNING,
      AgentState.FINISHED,
      AgentState.AWAITING_USER_CONFIRMATION,
      AgentState.USER_CONFIRMED,
      AgentState.USER_REJECTED,
    ].includes(agentState);

  // Display a yellow working icon while the runtime is starting
  if (
    conversationStatus === "STARTING" ||
    (!["STATUS$READY", null].includes(runtimeStatus) && !agentIsReady) ||
    (agentState != null &&
      [
        AgentState.LOADING,
        AgentState.PAUSED,
        AgentState.REJECTED,
        AgentState.RATE_LIMITED,
      ].includes(agentState))
  ) {
    return IndicatorColor.YELLOW;
  }

  if (agentState === AgentState.AWAITING_USER_CONFIRMATION) {
    return IndicatorColor.ORANGE;
  }

  if (agentState === AgentState.AWAITING_USER_INPUT) {
    return IndicatorColor.BLUE;
  }

  // All other agent states are green
  return IndicatorColor.GREEN;
}

export function getStatusCode(
  statusMessage: StatusMessage,
  webSocketStatus: V0_WebSocketStatus,
  conversationStatus: ConversationStatus | null,
  runtimeStatus: RuntimeStatus | null,
  agentState: AgentState | null,
  taskStatus?: V1AppConversationStartTaskStatus | null,
  subConversationTaskStatus?: V1AppConversationStartTaskStatus | null,
) {
  // PRIORITY 1: Handle task error state (when start-tasks API returns ERROR)
  // This must come first to prevent "Connecting..." from showing when task has errored
  if (taskStatus === "ERROR" || subConversationTaskStatus === "ERROR") {
    return I18nKey.AGENT_STATUS$ERROR_OCCURRED;
  }

  // PRIORITY 2: Handle conversation and runtime stopped states
  if (conversationStatus === "STOPPED" || runtimeStatus === "STATUS$STOPPED") {
    return I18nKey.CHAT_INTERFACE$STOPPED;
  }

  // Prioritize agent state when it indicates readiness, even if runtime status is stale
  const agentIsReady =
    agentState &&
    [
      AgentState.AWAITING_USER_INPUT,
      AgentState.RUNNING,
      AgentState.FINISHED,
      AgentState.PAUSED,
      AgentState.AWAITING_USER_CONFIRMATION,
      AgentState.USER_CONFIRMED,
      AgentState.USER_REJECTED,
    ].includes(agentState);

  if (
    runtimeStatus &&
    !["STATUS$READY", "STATUS$RUNTIME_STARTED"].includes(runtimeStatus) &&
    !agentIsReady // Skip runtime status check if agent is ready
  ) {
    const result = (I18nKey as { [key: string]: string })[runtimeStatus];
    if (result) {
      return result;
    }
    return runtimeStatus;
  }

  // PRIORITY 3: Handle WebSocket connection states
  // Note: WebSocket may be stuck in CONNECTING when task errors, so we check taskStatus first
  if (webSocketStatus === "DISCONNECTED") {
    return I18nKey.CHAT_INTERFACE$DISCONNECTED;
  }
  if (
    webSocketStatus === "CONNECTING" ||
    isTaskPolling(subConversationTaskStatus)
  ) {
    return I18nKey.CHAT_INTERFACE$CONNECTING;
  }

  // Handle agent states with simplified status messages
  if (agentState) {
    return AGENT_STATUS_MAP[agentState];
  }

  // Handle runtime status when no agent state
  if (runtimeStatus && runtimeStatus !== "STATUS$READY" && !agentState) {
    return runtimeStatus;
  }

  return I18nKey.CHAT_INTERFACE$AGENT_ERROR_MESSAGE;
}
