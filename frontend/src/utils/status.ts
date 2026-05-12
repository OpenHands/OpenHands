import { I18nKey } from "#/i18n/declaration";
import { AgentState } from "#/types/agent-state";
import { ConversationStatus } from "#/types/conversation-status";
import { RuntimeStatus } from "#/types/runtime-status";
import { V1AppConversationStartTaskStatus } from "#/api/conversation-service/v1-conversation-service.types";
import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";
import { V1ExecutionStatus } from "#/types/v1/core/base/common";
import { V1_WebSocketConnectionState } from "#/contexts/conversation-websocket-context";

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
  webSocketStatus: V1_WebSocketConnectionState,
  conversationStatus: ConversationStatus | null,
  runtimeStatus: RuntimeStatus | null,
  agentState: AgentState | null,
) {
  if (
    webSocketStatus === "CLOSED" ||
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
  webSocketConnectionState: V1_WebSocketConnectionState,
  executionStatus: V1ExecutionStatus | null,
  sandboxStatus: V1SandboxStatus | null,
  taskStatus?: V1AppConversationStartTaskStatus | null,
  subConversationTaskStatus?: V1AppConversationStartTaskStatus | null,
) {
  // TODO: The i18n keys in this method are scattered due to multiple iterations.
  // They should be unified under a single category

  // PRIORITY 1: Handle task error state (when start-tasks API returns ERROR)
  // This must come first to prevent "Connecting..." from showing when task has errored
  if (
    taskStatus === "ERROR" ||
    subConversationTaskStatus === "ERROR" ||
    sandboxStatus === "ERROR" ||
    executionStatus === "error"
  ) {
    return I18nKey.AGENT_STATUS$ERROR_OCCURRED;
  }

  // Priority 2 : Startup task.
  if (taskStatus && taskStatus !== "READY") {
    switch (taskStatus) {
      case "WAITING_FOR_SANDBOX":
        return I18nKey.COMMON$WAITING_FOR_SANDBOX;
      case "SETTING_UP_GIT_HOOKS":
        return I18nKey.STATUS$SETTING_UP_GIT_HOOKS;
      case "SETTING_UP_SKILLS":
        return I18nKey.STATUS$SETTING_UP_SKILLS;
      case "STARTING_CONVERSATION":
        return I18nKey.CONVERSATION$STARTING_CONVERSATION;
      case "WORKING":
      case "PREPARING_REPOSITORY":
      case "RUNNING_SETUP_SCRIPT":
        // TODO: These don't have In8n labels.
        return I18nKey.CONVERSATION$STARTING_CONVERSATION;
      default:
        throw new Error(`Unknown taskStatus: ${taskStatus}`);
    }
  }

  // Priority 3: SandboxStatus
  if (sandboxStatus && sandboxStatus !== "RUNNING") {
    switch (sandboxStatus) {
      case "STARTING":
        return I18nKey.CONVERSATION$STARTING_CONVERSATION;
      case "PAUSED":
      case "MISSING":
        return I18nKey.CHAT_INTERFACE$STOPPED;
      default:
        throw new Error(`Unknown sandboxStatus: ${sandboxStatus}`);
    }
  }

  // Websocket has disconnected...
  if (webSocketConnectionState && webSocketConnectionState !== "OPEN") {
    switch (webSocketConnectionState) {
      case "CLOSED":
      case "CLOSING":
        return I18nKey.CHAT_INTERFACE$DISCONNECTED;
      case "CONNECTING":
        return I18nKey.CHAT_INTERFACE$CONNECTING;
      default:
        throw new Error(
          `Unknown WebsocketConnectionState: ${webSocketConnectionState}`,
        );
    }
  }

  if (executionStatus && executionStatus !== V1ExecutionStatus.STUCK) {
    switch (executionStatus) {
      case V1ExecutionStatus.IDLE:
        return I18nKey.AGENT_STATUS$WAITING_FOR_TASK;
      case V1ExecutionStatus.RUNNING:
        return I18nKey.AGENT_STATUS$RUNNING_TASK;
      case V1ExecutionStatus.WAITING_FOR_CONFIRMATION:
        return I18nKey.AGENT_STATUS$WAITING_FOR_USER_CONFIRMATION;
      case V1ExecutionStatus.PAUSED:
        return I18nKey.CHAT_INTERFACE$AGENT_PAUSED_MESSAGE;
      case V1ExecutionStatus.FINISHED:
        return I18nKey.CHAT_INTERFACE$AGENT_FINISHED_MESSAGE;
      default:
        throw new Error(`Unknown executionStatus: ${executionStatus}`);
    }
  }

  // Some unknown state...
  return I18nKey.CHAT_INTERFACE$AGENT_ERROR_MESSAGE;
}
