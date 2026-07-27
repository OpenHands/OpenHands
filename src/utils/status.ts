import { I18nKey } from "#/i18n/declaration";
import { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { WebSocketConnectionState } from "#/contexts/conversation-websocket-context";

const ACTIVE_EXECUTION_STATUSES: ReadonlySet<ExecutionStatus> = new Set([
  ExecutionStatus.IDLE,
  ExecutionStatus.RUNNING,
  ExecutionStatus.WAITING_FOR_CONFIRMATION,
  ExecutionStatus.FINISHED,
]);

export function isExecutionActive(
  status: ExecutionStatus | null | undefined,
): boolean {
  return !!status && ACTIVE_EXECUTION_STATUSES.has(status);
}

export function isExecutionPaused(
  status: ExecutionStatus | null | undefined,
): boolean {
  return status === ExecutionStatus.PAUSED;
}

export function isExecutionErrored(
  status: ExecutionStatus | null | undefined,
): boolean {
  return status === ExecutionStatus.ERROR || status === ExecutionStatus.STUCK;
}

export function getTaskStatusI18nKey(
  taskStatus: AppConversationStartTaskStatus,
): I18nKey {
  switch (taskStatus) {
    case "WAITING_FOR_SANDBOX":
      return I18nKey.COMMON$WAITING_FOR_SANDBOX;
    case "SETTING_UP_GIT_HOOKS":
      return I18nKey.STATUS$SETTING_UP_GIT_HOOKS;
    case "SETTING_UP_SKILLS":
      return I18nKey.STATUS$SETTING_UP_SKILLS;
    // Terminal states map to their own localized keys so any caller that
    // delegates here (now or in the future) gets a correct label instead of
    // silently falling through to STARTING_CONVERSATION. Callers that need a
    // context-specific terminal label (e.g. getStatusCode's
    // AGENT_STATUS$ERROR_OCCURRED, or getStatusText's taskDetail precedence)
    // still handle these states before delegating.
    case "READY":
      return I18nKey.CONVERSATION$READY;
    case "ERROR":
      return I18nKey.COMMON$ERROR;
    // These collapse to the generic "Starting" label. `default` is unreachable
    // for the typed union but is kept as a runtime safety net: the start-task
    // API may report a new status before this enum is updated, in which case we
    // degrade to "Starting" rather than throwing (see FUTURE_STATUS_FROM_CLOUD).
    case "STARTING_CONVERSATION":
    case "WORKING":
    case "PREPARING_REPOSITORY":
    case "RUNNING_SETUP_SCRIPT":
    default:
      return I18nKey.CONVERSATION$STARTING_CONVERSATION;
  }
}

export function getStatusCode(
  webSocketConnectionState: WebSocketConnectionState,
  executionStatus: ExecutionStatus | null,
  taskStatus?: AppConversationStartTaskStatus | null,
  subConversationTaskStatus?: AppConversationStartTaskStatus | null,
) {
  if (
    taskStatus === "ERROR" ||
    subConversationTaskStatus === "ERROR" ||
    executionStatus === "error"
  ) {
    return I18nKey.AGENT_STATUS$ERROR_OCCURRED;
  }

  if (taskStatus && taskStatus !== "READY") {
    return getTaskStatusI18nKey(taskStatus);
  }

  if (executionStatus === ExecutionStatus.PAUSED) {
    return I18nKey.CHAT_INTERFACE$STOPPED;
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

  if (executionStatus && executionStatus !== ExecutionStatus.STUCK) {
    switch (executionStatus) {
      case ExecutionStatus.IDLE:
        return I18nKey.AGENT_STATUS$WAITING_FOR_TASK;
      case ExecutionStatus.RUNNING:
        return I18nKey.AGENT_STATUS$RUNNING_TASK;
      case ExecutionStatus.WAITING_FOR_CONFIRMATION:
        return I18nKey.AGENT_STATUS$WAITING_FOR_USER_CONFIRMATION;
      case ExecutionStatus.FINISHED:
        return I18nKey.CHAT_INTERFACE$AGENT_FINISHED_MESSAGE;
      default:
        throw new Error(`Unknown executionStatus: ${executionStatus}`);
    }
  }

  return I18nKey.CHAT_INTERFACE$AGENT_ERROR_MESSAGE;
}
