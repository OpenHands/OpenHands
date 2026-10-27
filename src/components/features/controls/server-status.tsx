import { useTranslation } from "react-i18next";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { getStatusColor, getStatusText } from "#/utils/utils";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { isExecutionErrored, isExecutionPaused } from "#/utils/status";
import { I18nKey } from "#/i18n/declaration";
import { ConversationStatusDot } from "#/components/features/conversation-panel/conversation-status-dot";

export interface ServerStatusProps {
  className?: string;
  executionStatus: ExecutionStatus | null;
  isPausing?: boolean;
  /**
   * The runtime is missing and cannot be resumed. Shows the same archive
   * indicator and "Archived" label as the sidebar row instead of a status
   * derived from the (stale) execution status.
   */
  isArchived?: boolean;
}

export function ServerStatus({
  className = "",
  executionStatus,
  isPausing = false,
  isArchived = false,
}: ServerStatusProps) {
  const { curAgentState } = useAgentState();
  const { isTask, taskStatus, taskDetail } = useTaskPolling();
  const { t } = useTranslation("openhands");
  const { errorMessage } = useErrorMessageStore();

  const isStartingStatus =
    curAgentState === AgentState.LOADING || curAgentState === AgentState.INIT;
  const isStopStatus =
    isExecutionErrored(executionStatus) || isExecutionPaused(executionStatus);

  const statusColor = getStatusColor({
    isPausing,
    isTask,
    taskStatus,
    isStartingStatus,
    isStopStatus,
    curAgentState,
  });

  const statusText = getStatusText({
    isPausing,
    isTask,
    taskStatus,
    taskDetail,
    isStartingStatus,
    isStopStatus,
    curAgentState,
    errorMessage,
    t,
  });

  return (
    <div className={className} data-testid="server-status">
      <div className="flex items-center">
        {isArchived ? (
          <span className="w-6 h-6 shrink-0 flex items-center justify-center">
            <ConversationStatusDot
              executionStatus={executionStatus}
              isArchived
              showTooltip={false}
            />
          </span>
        ) : (
          <DebugStackframeDot
            className="w-6 h-6 shrink-0"
            color={statusColor}
          />
        )}
        <span className="text-[13px] text-contrast font-normal">
          {isArchived ? t(I18nKey.COMMON$ARCHIVED) : statusText}
        </span>
      </div>
    </div>
  );
}

export default ServerStatus;
