import { useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AgentState } from "#/types/agent-state";
import { ActionTooltip } from "../action-tooltip";
import { RiskAlert } from "#/components/shared/risk-alert";
import WarningIcon from "#/icons/u-warning.svg?react";
import { useEventMessageStore } from "#/stores/event-message-store";
import { useEventStore } from "#/stores/use-event-store";
import { isV1Event, isActionEvent } from "#/types/v1/type-guards";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useAgentState } from "#/hooks/use-agent-state";
import { useRespondToConfirmation } from "#/hooks/mutation/use-respond-to-confirmation";
import { SecurityRisk } from "#/types/v1/core/base/common";

export function V1ConfirmationButtons() {
  const v1SubmittedEventIds = useEventMessageStore(
    (state) => state.v1SubmittedEventIds,
  );
  const addV1SubmittedEventId = useEventMessageStore(
    (state) => state.addV1SubmittedEventId,
  );

  const { t } = useTranslation();
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const { mutate: respondToConfirmation } = useRespondToConfirmation();
  const events = useEventStore((state) => state.events);

  // Find the most recent V1 action awaiting confirmation
  const awaitingAction = events
    .filter(isV1Event)
    .slice()
    .reverse()
    .find((ev) => {
      if (ev.source !== "agent") return false;
      // For V1, we check if the agent state is waiting for confirmation
      return curAgentState === AgentState.AWAITING_USER_CONFIRMATION;
    });

  const handleConfirmation = useCallback(
    (accept: boolean) => {
      if (!awaitingAction || !conversation) {
        return;
      }

      // Mark event as submitted to prevent duplicate submissions
      addV1SubmittedEventId(awaitingAction.id);

      // Call the V1 API endpoint
      respondToConfirmation({
        conversationId: conversation.conversation_id,
        conversationUrl: conversation.url || "",
        sessionApiKey: conversation.session_api_key,
        accept,
      });
    },
    [
      awaitingAction,
      conversation,
      addV1SubmittedEventId,
      respondToConfirmation,
    ],
  );

  // Handle keyboard shortcuts
  useEffect(() => {
    if (!awaitingAction) {
      return undefined;
    }

    const handleCancelShortcut = (event: KeyboardEvent) => {
      if (event.shiftKey && event.metaKey && event.key === "Backspace") {
        event.preventDefault();
        handleConfirmation(false);
      }
    };

    const handleContinueShortcut = (event: KeyboardEvent) => {
      if (event.metaKey && event.key === "Enter") {
        event.preventDefault();
        handleConfirmation(true);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      // Cancel: Shift+Cmd+Backspace (⇧⌘⌫)
      handleCancelShortcut(event);
      // Continue: Cmd+Enter (⌘↩)
      handleContinueShortcut(event);
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [awaitingAction, handleConfirmation]);

  // Only show if agent is waiting for confirmation and we haven't already submitted
  if (
    curAgentState !== AgentState.AWAITING_USER_CONFIRMATION ||
    !awaitingAction ||
    v1SubmittedEventIds.includes(awaitingAction.id)
  ) {
    return null;
  }

  // Get security risk from the action (only ActionEvent has security_risk)
  const risk = isActionEvent(awaitingAction)
    ? awaitingAction.security_risk
    : SecurityRisk.UNKNOWN;

  const isHighRisk = risk === SecurityRisk.HIGH;

  return (
    <div className="flex flex-col gap-2 pt-4">
      {isHighRisk && (
        <RiskAlert
          content={t(I18nKey.CHAT_INTERFACE$HIGH_RISK_WARNING)}
          icon={<WarningIcon width={16} height={16} color="#fff" />}
          severity="high"
          title={t(I18nKey.COMMON$HIGH_RISK)}
        />
      )}
      <div className="flex justify-between items-center">
        <p className="text-sm font-normal text-white">
          {t(I18nKey.CHAT_INTERFACE$USER_ASK_CONFIRMATION)}
        </p>
        <div className="flex items-center gap-3">
          <ActionTooltip
            type="reject"
            onClick={() => handleConfirmation(false)}
          />
          <ActionTooltip
            type="confirm"
            onClick={() => handleConfirmation(true)}
          />
        </div>
      </div>
    </div>
  );
}
