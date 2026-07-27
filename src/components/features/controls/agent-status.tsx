import { useTranslation } from "react-i18next";
import { useEffect, useState } from "react";
import { CircleCheck } from "lucide-react";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { getStatusCode } from "#/utils/status";
import { ChatStopButton } from "../chat/chat-stop-button";
import { AgentState } from "#/types/agent-state";
import ClockIcon from "#/icons/u-clock-three.svg?react";
import { ChatResumeAgentButton } from "../chat/chat-play-button";
import { cn, isTaskPolling } from "#/utils/utils";
import { AgentLoading } from "./agent-loading";
import { useConversationStore } from "#/stores/conversation-store";
import CircleErrorIcon from "#/icons/circle-error.svg?react";
import { useAgentState } from "#/hooks/use-agent-state";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import { useAgentNotification } from "#/hooks/use-agent-notification";
import { I18nKey } from "#/i18n/declaration";

export interface AgentStatusProps {
  className?: string;
  handleStop: () => void;
  handleResumeAgent: () => void;
  disabled?: boolean;
  isPausing?: boolean;
}

export function AgentStatus({
  className = "",
  handleStop,
  handleResumeAgent,
  disabled = false,
  isPausing = false,
}: AgentStatusProps) {
  const { t } = useTranslation("openhands");
  const { setShouldShownAgentLoading } = useConversationStore();
  const { curAgentState, executionStatus } = useAgentState();

  // Trigger browser tab flash and notification sound on state changes
  useAgentNotification(curAgentState);
  const webSocketStatus = useUnifiedWebSocketStatus();
  const { data: conversation } = useActiveConversation();
  const { taskStatus } = useTaskPolling();

  const { subConversationTaskId } = useConversationStore();

  // Poll sub-conversation task to track its loading state
  const { taskStatus: subConversationTaskStatus } =
    useSubConversationTaskPolling(
      subConversationTaskId,
      conversation?.id || null,
    );

  const statusCode = getStatusCode(
    webSocketStatus,
    executionStatus ?? null,
    taskStatus,
    subConversationTaskStatus,
  );

  const shouldShownAgentLoading =
    curAgentState === AgentState.INIT ||
    curAgentState === AgentState.LOADING ||
    (webSocketStatus === "CONNECTING" && taskStatus !== "ERROR") ||
    isTaskPolling(taskStatus) ||
    isTaskPolling(subConversationTaskStatus);

  // For UI rendering - includes pause state
  const isLoading = shouldShownAgentLoading || isPausing;

  const shouldShownAgentError =
    curAgentState === AgentState.ERROR ||
    curAgentState === AgentState.RATE_LIMITED ||
    webSocketStatus === "CLOSED" ||
    taskStatus === "ERROR";

  const shouldShownAgentStop =
    !shouldShownAgentError && curAgentState === AgentState.RUNNING;

  const shouldShownAgentResume =
    !shouldShownAgentError &&
    (curAgentState === AgentState.STOPPED ||
      curAgentState === AgentState.PAUSED);
  const isInteractive =
    !isLoading && (shouldShownAgentStop || shouldShownAgentResume);
  const isDoneStatus =
    statusCode === I18nKey.CHAT_INTERFACE$AGENT_FINISHED_MESSAGE;
  const isReadyStatus = statusCode === I18nKey.AGENT_STATUS$WAITING_FOR_TASK;
  const isTransientCheckStatus = isDoneStatus || isReadyStatus;
  const [shouldRenderDoneStatus, setShouldRenderDoneStatus] = useState(true);
  const [shouldFadeDoneStatus, setShouldFadeDoneStatus] = useState(false);

  // Update global state when agent loading condition changes
  useEffect(() => {
    setShouldShownAgentLoading(!!shouldShownAgentLoading);
  }, [shouldShownAgentLoading, setShouldShownAgentLoading]);

  useEffect(() => {
    if (!isTransientCheckStatus) {
      setShouldRenderDoneStatus(true);
      setShouldFadeDoneStatus(false);
      return;
    }

    setShouldRenderDoneStatus(true);
    setShouldFadeDoneStatus(false);

    const fadeTimer = window.setTimeout(() => {
      setShouldFadeDoneStatus(true);
    }, 1000);

    const hideTimer = window.setTimeout(() => {
      setShouldRenderDoneStatus(false);
    }, 1500);

    return () => {
      window.clearTimeout(fadeTimer);
      window.clearTimeout(hideTimer);
    };
  }, [isTransientCheckStatus]);

  if (isTransientCheckStatus && !shouldRenderDoneStatus) {
    return null;
  }

  return (
    <div
      className={cn(
        "flex items-center gap-1 min-w-0",
        isTransientCheckStatus && "transition-opacity duration-500",
        shouldFadeDoneStatus && "opacity-0",
        className,
      )}
    >
      <span
        className="text-[11px] text-[var(--oh-muted)] font-normal leading-5 min-w-0 max-w-full truncate"
        title={t(statusCode)}
      >
        {t(statusCode)}
      </span>
      <div
        className={cn(
          "box-border content-stretch flex flex-row gap-[3px] items-center justify-center overflow-clip px-0.5 py-1 relative rounded-[100px] shrink-0 size-6 transition-all duration-200 active:scale-95 bg-transparent text-[var(--oh-muted)] hover:bg-white/10 hover:text-white",
          isInteractive ? "cursor-pointer" : "cursor-default",
        )}
      >
        {isLoading && <AgentLoading />}
        {!isLoading && shouldShownAgentStop && (
          <ChatStopButton handleStop={handleStop} />
        )}
        {!isLoading && shouldShownAgentResume && (
          <ChatResumeAgentButton
            onAgentResumed={handleResumeAgent}
            disabled={disabled}
          />
        )}
        {!isLoading && shouldShownAgentError && (
          <CircleErrorIcon
            className="w-4 h-4 text-current"
            data-testid="circle-error-icon"
          />
        )}
        {!isLoading &&
          !shouldShownAgentStop &&
          !shouldShownAgentResume &&
          !shouldShownAgentError &&
          (isTransientCheckStatus ? (
            <CircleCheck className="w-4 h-4 text-current" />
          ) : (
            <ClockIcon className="w-4 h-4 text-current" />
          ))}
      </div>
    </div>
  );
}

export default AgentStatus;
