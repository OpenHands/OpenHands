import { useTranslation } from "react-i18next";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { getStatusColor, getStatusText } from "#/utils/utils";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { isExecutionErrored, isExecutionPaused } from "#/utils/status";

export interface ServerStatusProps {
  className?: string;
  executionStatus: ExecutionStatus | null;
  isPausing?: boolean;
}

export function ServerStatus({
  className = "",
  executionStatus,
  isPausing = false,
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
        <DebugStackframeDot className="w-6 h-6 shrink-0" color={statusColor} />
        <span className="text-[13px] text-white font-normal">{statusText}</span>
      </div>
    </div>
  );
}

export default ServerStatus;
