import React, { useMemo, useEffect, useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import ChevronDownSmallIcon from "#/icons/chevron-down-small.svg?react";
import LessonPlanIcon from "#/icons/lesson-plan.svg?react";
import { useConversationStore } from "#/stores/conversation-store";
import { ChangeAgentContextMenu } from "./change-agent-context-menu";
import { cn } from "#/utils/utils";
import { USE_PLANNING_AGENT } from "#/utils/feature-flags";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";

export function ChangeAgentButton() {
  const [contextMenuOpen, setContextMenuOpen] = useState<boolean>(false);

  const { conversationMode, setConversationMode, subConversationTaskId } =
    useConversationStore();

  const webSocketStatus = useUnifiedWebSocketStatus();

  const isWebSocketConnected = webSocketStatus === "CONNECTED";

  const shouldUsePlanningAgent = USE_PLANNING_AGENT();

  const { curAgentState } = useAgentState();

  const { t } = useTranslation();

  const isAgentRunning = curAgentState === AgentState.RUNNING;

  const { data: conversation } = useActiveConversation();

  const queryClient = useQueryClient();

  // Track the last invalidated task ID to prevent duplicate invalidations
  const lastInvalidatedTaskIdRef = useRef<string | null>(null);

  // Poll sub-conversation task status
  const { taskStatus, subConversationId } = useSubConversationTaskPolling(
    subConversationTaskId,
    conversation?.conversation_id || null,
  );

  // Invalidate parent conversation cache when task is ready (only once per task)
  useEffect(() => {
    if (
      taskStatus === "READY" &&
      subConversationId &&
      conversation?.conversation_id &&
      subConversationTaskId &&
      lastInvalidatedTaskIdRef.current !== subConversationTaskId
    ) {
      // Mark this task as invalidated to prevent duplicate calls
      lastInvalidatedTaskIdRef.current = subConversationTaskId;
      // Invalidate the parent conversation to refetch with updated sub_conversation_ids
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", conversation.conversation_id],
      });
    }
  }, [
    taskStatus,
    subConversationId,
    conversation?.conversation_id,
    subConversationTaskId,
    queryClient,
  ]);

  // Get handlePlanClick and isCreatingConversation from custom hook
  const { handlePlanClick, isCreatingConversation } = useHandlePlanClick();

  // Close context menu when agent starts running
  useEffect(() => {
    if ((isAgentRunning || !isWebSocketConnected) && contextMenuOpen) {
      setContextMenuOpen(false);
    }
  }, [isAgentRunning, contextMenuOpen, isWebSocketConnected]);

  const isButtonDisabled =
    isAgentRunning ||
    isCreatingConversation ||
    !isWebSocketConnected ||
    !shouldUsePlanningAgent;

  // Handle Shift + Tab keyboard shortcut to cycle through modes
  useEffect(() => {
    if (isButtonDisabled) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      // Check for Shift + Tab combination
      if (event.shiftKey && event.key === "Tab") {
        // Prevent default tab navigation behavior
        event.preventDefault();
        event.stopPropagation();

        // Cycle between modes: code -> plan -> code
        const nextMode = conversationMode === "code" ? "plan" : "code";
        if (nextMode === "plan") {
          handlePlanClick(event);
        } else {
          setConversationMode(nextMode);
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [
    isButtonDisabled,
    conversationMode,
    setConversationMode,
    handlePlanClick,
  ]);

  const handleButtonClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setContextMenuOpen(!contextMenuOpen);
  };

  const handleCodeClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setConversationMode("code");
  };

  const isExecutionAgent = conversationMode === "code";

  const buttonLabel = useMemo(() => {
    if (isExecutionAgent) {
      return t(I18nKey.COMMON$CODE);
    }
    return t(I18nKey.COMMON$PLAN);
  }, [isExecutionAgent, t]);

  const buttonIcon = useMemo(() => {
    if (isExecutionAgent) {
      return <CodeTagIcon width={18} height={18} color="#737373" />;
    }
    return <LessonPlanIcon width={18} height={18} color="#ffffff" />;
  }, [isExecutionAgent]);

  if (!shouldUsePlanningAgent) {
    return null;
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={handleButtonClick}
        disabled={isButtonDisabled}
        className={cn(
          "flex items-center border border-[#4B505F] rounded-[100px] transition-opacity",
          !isExecutionAgent && "border-[#597FF4] bg-[#4A67BD]",
          isButtonDisabled
            ? "opacity-50 cursor-not-allowed"
            : "cursor-pointer hover:opacity-80",
        )}
      >
        <div className="flex items-center gap-1 pl-1.5">
          {buttonIcon}
          <Typography.Text className="text-white text-2.75 not-italic font-normal leading-5">
            {buttonLabel}
          </Typography.Text>
        </div>
        <ChevronDownSmallIcon width={24} height={24} color="#ffffff" />
      </button>
      {contextMenuOpen && (
        <ChangeAgentContextMenu
          onClose={() => setContextMenuOpen(false)}
          onCodeClick={handleCodeClick}
          onPlanClick={handlePlanClick}
        />
      )}
    </div>
  );
}
