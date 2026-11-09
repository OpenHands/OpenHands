import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { OH_STATUS_ERROR_COLOR } from "#/constants/status-colors";
import { useTranslation } from "react-i18next";
import { useAgentState } from "#/hooks/use-agent-state";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { getStatusColor, getStatusText } from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { ConversationName } from "./conversation-name";
import { ConversationGitActionsToggle } from "./conversation-git-actions-toggle";
import { ConversationOverviewToggle } from "./conversation-overview-toggle";
import { RightPanelToggle } from "./right-panel-toggle";
import { isExecutionErrored, isExecutionPaused } from "#/utils/status";
import { I18nKey } from "#/i18n/declaration";

/**
 * Conversation title row. The dot left of the title only shows status;
 * stopping is in the composer (agent) and the title menu (sandbox).
 */
export function ConversationNameWithStatus() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const launchError = useConversationLaunchStore((state) =>
    conversationId ? state.launches[conversationId]?.error : undefined,
  );
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const { isTask, taskStatus, taskDetail } = useTaskPolling();
  const { errorMessage } = useErrorMessageStore();

  const executionStatus = conversation?.execution_status ?? null;
  const isStartingStatus =
    !conversation ||
    curAgentState === AgentState.LOADING ||
    curAgentState === AgentState.INIT;
  const isErrored = Boolean(launchError) || isExecutionErrored(executionStatus);

  const statusColor = launchError
    ? OH_STATUS_ERROR_COLOR
    : getStatusColor({
        isPausing: false,
        isTask,
        taskStatus,
        isStartingStatus,
        isStopStatus: isExecutionErrored(executionStatus),
        curAgentState,
      });

  const statusText = isErrored
    ? errorMessage || t(I18nKey.COMMON$ERROR)
    : getStatusText({
        isPausing: false,
        isTask,
        taskStatus,
        taskDetail,
        isStartingStatus,
        isStopStatus: isExecutionPaused(executionStatus),
        curAgentState,
        errorMessage,
        t,
      });

  return (
    <div className="flex items-center justify-between w-full">
      <div className="flex items-center min-w-0">
        <StyledTooltip content={statusText} placement="bottom">
          <span
            role="status"
            aria-label={statusText}
            data-testid="conversation-status"
            className="flex shrink-0 items-center justify-center rounded-md"
          >
            <DebugStackframeDot
              className="ml-[3.5px] w-6 h-6"
              color={statusColor}
              aria-hidden
            />
          </span>
        </StyledTooltip>
        <ConversationName />
      </div>
      <div className="mr-2 flex shrink-0 items-center gap-1">
        <ConversationGitActionsToggle />
        <ConversationOverviewToggle />
        <RightPanelToggle />
      </div>
    </div>
  );
}
