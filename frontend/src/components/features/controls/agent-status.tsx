import { useTranslation } from "react-i18next";
import { useEffect } from "react";
import { useStatusStore } from "#/state/status-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { getStatusCode } from "#/utils/status";
import { ChatStopButton } from "../chat/chat-stop-button";
import { AgentState } from "#/types/agent-state";
import ClockIcon from "#/icons/u-clock-three.svg?react";
import { ChatResumeAgentButton } from "../chat/chat-play-button";
import { cn, isTaskPolling } from "#/utils/utils";
import { AgentLoading } from "./agent-loading";
import { useConversationStore } from "#/state/conversation-store";
import CircleErrorIcon from "#/icons/circle-error.svg?react";
import { useAgentState } from "#/hooks/use-agent-state";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";

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
  const { t } = useTranslation();
  const { setShouldShownAgentLoading } = useConversationStore();
  const { curAgentState } = useAgentState();
  const { curStatusMessage } = useStatusStore();
  const webSocketStatus = useUnifiedWebSocketStatus();
  const { data: conversation } = useActiveConversation();
  const { taskStatus } = useTaskPolling();

  const { subConversationTaskId } = useConversationStore();

  // Poll sub-conversation task to track its loading state
  const { taskStatus: subConversationTaskStatus } =
    useSubConversationTaskPolling(
      subConversationTaskId,
      conversation?.conversation_id || null,
    );

  const statusCode = getStatusCode(
    curStatusMessage,
    webSocketStatus,
    conversation?.status || null,
    conversation?.runtime_status || null,
    curAgentState,
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
    webSocketStatus === "DISCONNECTED" ||
    taskStatus === "ERROR";

  const shouldShownAgentStop = curAgentState === AgentState.RUNNING;

  const shouldShownAgentResume =
    curAgentState === AgentState.STOPPED || curAgentState === AgentState.PAUSED;

  // Update global state when agent loading condition changes
  useEffect(() => {
    setShouldShownAgentLoading(!!shouldShownAgentLoading);
  }, [shouldShownAgentLoading, setShouldShownAgentLoading]);

  return (
    <div className={cn("flex items-center gap-1 min-w-0", className)}>
      <span
        className="text-[11px] text-white font-normal leading-5 flex-1 min-w-0 max-w-full whitespace-normal break-words"
        title={t(statusCode)}
      >
        {t(statusCode)}
      </span>
      <div
        className={cn(
          "bg-[#525252] box-border content-stretch flex flex-row gap-[3px] items-center justify-center overflow-clip px-0.5 py-1 relative rounded-[100px] shrink-0 size-6 transition-all duration-200 active:scale-95",
          !isLoading &&
            (shouldShownAgentStop || shouldShownAgentResume) &&
            "hover:bg-[#737373] cursor-pointer",
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
            className="w-4 h-4"
            data-testid="circle-error-icon"
          />
        )}
        {!isLoading &&
          !shouldShownAgentStop &&
          !shouldShownAgentResume &&
          !shouldShownAgentError && <ClockIcon className="w-4 h-4" />}
      </div>
    </div>
  );
}

export default AgentStatus;
