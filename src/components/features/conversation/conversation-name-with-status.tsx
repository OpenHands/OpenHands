import React from "react";
import { useAgentState } from "#/hooks/use-agent-state";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUnifiedPauseConversation } from "#/hooks/mutation/use-unified-stop-conversation";
import { useUnifiedResumeConversation } from "#/hooks/mutation/use-unified-start-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useUserProviders } from "#/hooks/use-user-providers";
import { getStatusColor } from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";
import { ServerStatusContextMenu } from "../controls/server-status-context-menu";
import { ConversationName } from "./conversation-name";
import { RightPanelToggle } from "./right-panel-toggle";
import {
  isExecutionActive,
  isExecutionErrored,
  isExecutionPaused,
} from "#/utils/status";

export function ConversationNameWithStatus() {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const { isTask, taskStatus } = useTaskPolling();
  const { mutate: pauseConversation } = useUnifiedPauseConversation();
  const { mutate: resumeConversation } = useUnifiedResumeConversation();
  const { providers } = useUserProviders();

  const executionStatus = conversation?.execution_status ?? null;
  const isStartingStatus =
    curAgentState === AgentState.LOADING || curAgentState === AgentState.INIT;
  const isStopStatus = isExecutionErrored(executionStatus);

  const statusColor = getStatusColor({
    isPausing: false,
    isTask,
    taskStatus,
    isStartingStatus,
    isStopStatus,
    curAgentState,
  });

  const handleStopServer = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (conversationId) {
      pauseConversation({ conversationId });
    }
  };

  const handleStartServer = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (conversationId) {
      resumeConversation({ conversationId, providers });
    }
  };

  return (
    <div className="flex items-center justify-between w-full">
      <div className="flex items-center min-w-0">
        <div className="group relative shrink-0">
          <DebugStackframeDot
            className="ml-[3.5px] w-6 h-6 cursor-pointer"
            color={statusColor}
          />
          <ServerStatusContextMenu
            onClose={() => {}}
            onStopServer={
              isExecutionActive(executionStatus) ? handleStopServer : undefined
            }
            onStartServer={
              isExecutionPaused(executionStatus) ? handleStartServer : undefined
            }
            executionStatus={executionStatus}
            position="bottom"
            className="opacity-0 invisible pointer-events-none group-hover:opacity-100 group-hover:visible group-hover:pointer-events-auto bottom-full left-0 mt-0 min-h-fit"
            isPausing={false}
          />
        </div>
        <ConversationName />
      </div>
      <RightPanelToggle className="mr-2" />
    </div>
  );
}
