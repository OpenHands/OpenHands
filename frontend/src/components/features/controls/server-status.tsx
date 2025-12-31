import { useTranslation } from "react-i18next";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { ConversationStatus } from "#/types/conversation-status";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { getStatusColor } from "#/utils/utils";
import { useErrorMessageStore } from "#/stores/error-message-store";

export interface ServerStatusProps {
  className?: string;
  conversationStatus: ConversationStatus | null;
  isPausing?: boolean;
}

export function ServerStatus({
  className = "",
  conversationStatus,
  isPausing = false,
}: ServerStatusProps) {
  const { curAgentState } = useAgentState();
  const { t } = useTranslation();
  const { isTask, taskStatus, taskDetail } = useTaskPolling();
  const { errorMessage } = useErrorMessageStore();

  const isStartingStatus =
    curAgentState === AgentState.LOADING || curAgentState === AgentState.INIT;

  const isStopStatus = conversationStatus === "STOPPED";

  const statusColor = getStatusColor({
    isPausing,
    isTask,
    taskStatus,
    isStartingStatus,
    isStopStatus,
    curAgentState,
  });

  const getStatusText = (): string => {
    // Show pausing status
    if (isPausing) {
      return t(I18nKey.COMMON$STOPPING);
    }

    // Show task status if we're polling a task
    if (isTask && taskStatus) {
      if (taskStatus === "ERROR") {
        return (
          taskDetail || t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION)
        );
      }
      if (taskStatus === "READY") {
        return t(I18nKey.CONVERSATION$READY);
      }
      // Format status text: "WAITING_FOR_SANDBOX" -> "Waiting for sandbox"
      return (
        taskDetail ||
        taskStatus
          .toLowerCase()
          .replace(/_/g, " ")
          .replace(/\b\w/g, (c) => c.toUpperCase())
      );
    }

    if (isStartingStatus) {
      return t(I18nKey.COMMON$STARTING);
    }
    if (isStopStatus) {
      return t(I18nKey.COMMON$SERVER_STOPPED);
    }
    if (curAgentState === AgentState.ERROR) {
      return errorMessage || t(I18nKey.COMMON$ERROR);
    }
    return t(I18nKey.COMMON$RUNNING);
  };

  const statusText = getStatusText();

  return (
    <div className={className} data-testid="server-status">
      <div className="flex items-center">
        <DebugStackframeDot className="w-6 h-6 shrink-0" color={statusColor} />
        <span className="text-[13px] text-white font-normal">{statusText}</span>
      </div>
    </div>
  );
}

export default ServerStatus;
