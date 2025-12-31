import React from "react";
import { useParams } from "react-router";
import { useAgentState } from "#/hooks/use-agent-state";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUnifiedPauseConversationSandbox } from "#/hooks/mutation/use-unified-stop-conversation";
import { useUnifiedResumeConversationSandbox } from "#/hooks/mutation/use-unified-start-conversation";
import { useUserProviders } from "#/hooks/use-user-providers";
import { getStatusColor } from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";
import { ServerStatusContextMenu } from "../controls/server-status-context-menu";
import { ConversationName } from "./conversation-name";

export function ConversationNameWithStatus() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const { isTask, taskStatus } = useTaskPolling();
  const { mutate: pauseConversationSandbox } =
    useUnifiedPauseConversationSandbox();
  const { mutate: resumeConversationSandbox } =
    useUnifiedResumeConversationSandbox();
  const { providers } = useUserProviders();

  const isStartingStatus =
    curAgentState === AgentState.LOADING || curAgentState === AgentState.INIT;
  const isStopStatus = conversation?.status === "STOPPED";

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
      pauseConversationSandbox({ conversationId });
    }
  };

  const handleStartServer = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (conversationId) {
      resumeConversationSandbox({ conversationId, providers });
    }
  };

  return (
    <div className="flex items-center">
      <div className="group relative">
        <DebugStackframeDot
          className="ml-[3.5px] w-6 h-6 cursor-pointer"
          color={statusColor}
        />
        <ServerStatusContextMenu
          onClose={() => {}}
          onStopServer={
            conversation?.status === "RUNNING" ? handleStopServer : undefined
          }
          onStartServer={
            conversation?.status === "STOPPED" ? handleStartServer : undefined
          }
          conversationStatus={conversation?.status ?? null}
          position="bottom"
          className="opacity-0 invisible pointer-events-none group-hover:opacity-100 group-hover:visible group-hover:pointer-events-auto bottom-full left-0 mt-0 min-h-fit"
          isPausing={false}
        />
      </div>
      <ConversationName />
    </div>
  );
}
