import React, { useMemo, useEffect, useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { ComboboxCaretInline } from "#/ui/combobox-caret";
import LessonPlanIcon from "#/icons/lesson-plan.svg?react";
import { CodePillIcon } from "#/icons/code-pill";
import { useConversationStore } from "#/stores/conversation-store";
import { ChangeAgentContextMenu } from "./change-agent-context-menu";
import { cn } from "#/utils/utils";
import {
  formControlMutedHoverClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUnifiedWebSocketStatus } from "#/hooks/use-unified-websocket-status";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

export function ChangeAgentButton() {
  const [contextMenuOpen, setContextMenuOpen] = useState<boolean>(false);

  const { conversationMode, setConversationMode, subConversationTaskId } =
    useConversationStore();

  const { conversationId } = useOptionalConversationId();

  const isHomePage = !conversationId;

  const webSocketStatus = useUnifiedWebSocketStatus();

  const isWebSocketConnected = webSocketStatus === "OPEN";

  const { curAgentState } = useAgentState();

  const { t } = useTranslation("openhands");

  const isAgentRunning = curAgentState === AgentState.RUNNING;

  const { data: conversation } = useActiveConversation();

  const queryClient = useQueryClient();

  // Track the last invalidated task ID to prevent duplicate invalidations
  const lastInvalidatedTaskIdRef = useRef<string | null>(null);

  // Poll sub-conversation task status
  const { taskStatus, subConversationId } = useSubConversationTaskPolling(
    subConversationTaskId,
    conversation?.id || null,
  );

  // Invalidate parent conversation cache when task is ready (only once per task)
  useEffect(() => {
    if (
      taskStatus === "READY" &&
      subConversationId &&
      conversation?.id &&
      subConversationTaskId &&
      lastInvalidatedTaskIdRef.current !== subConversationTaskId
    ) {
      // Mark this task as invalidated to prevent duplicate calls
      lastInvalidatedTaskIdRef.current = subConversationTaskId;
      // Invalidate the parent conversation to refetch with updated sub_conversation_ids
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", conversation.id],
      });
    }
  }, [
    taskStatus,
    subConversationId,
    conversation?.id,
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
    isHomePage ||
    isAgentRunning ||
    isCreatingConversation ||
    !isWebSocketConnected;

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
      return <CodePillIcon className="h-[11px] w-[11px] shrink-0" />;
    }
    return <LessonPlanIcon width={18} height={18} color="currentColor" />;
  }, [isExecutionAgent]);

  const button = (
    <div className="relative">
      <button
        type="button"
        onClick={handleButtonClick}
        disabled={isButtonDisabled}
        className={cn(
          "flex items-center rounded-[100px]",
          formControlTransitionClassName,
          isExecutionAgent
            ? "border border-transparent text-[var(--oh-muted)]"
            : "border border-[#597FF4] bg-[#4A67BD]",
          !isButtonDisabled &&
            isExecutionAgent &&
            cn("cursor-pointer", formControlMutedHoverClassName),
          !isButtonDisabled &&
            !isExecutionAgent &&
            "cursor-pointer text-white hover:bg-[#597FF4]",
          isButtonDisabled &&
            cn(
              "opacity-50 cursor-not-allowed",
              isExecutionAgent && "border-transparent",
            ),
        )}
      >
        <div className="flex items-center gap-1 pl-1.5">
          {buttonIcon}
          <Typography.Text className="text-2.75 not-italic font-normal leading-5">
            {buttonLabel}
          </Typography.Text>
        </div>
        <ComboboxCaretInline isOpen={contextMenuOpen} />
      </button>
      {contextMenuOpen && (
        <ChangeAgentContextMenu
          activeMode={conversationMode}
          onClose={() => setContextMenuOpen(false)}
          onCodeClick={handleCodeClick}
          onPlanClick={handlePlanClick}
        />
      )}
    </div>
  );

  if (isHomePage) {
    return (
      <StyledTooltip
        content={t(I18nKey.CHANGE_AGENT$SWITCH_AFTER_CONVERSATION)}
        placement="top"
      >
        {button}
      </StyledTooltip>
    );
  }

  return button;
}
