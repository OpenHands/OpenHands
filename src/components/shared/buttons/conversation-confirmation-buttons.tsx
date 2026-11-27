import { useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AgentState } from "#/types/agent-state";
import { ActionTooltip } from "../action-tooltip";
import { RiskAlert } from "#/components/shared/risk-alert";
import WarningIcon from "#/icons/u-warning.svg?react";
import { useEventMessageStore } from "#/stores/event-message-store";
import { useEventStore } from "#/stores/use-event-store";
import { isActionEvent } from "#/types/agent-server/type-guards";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useAgentState } from "#/hooks/use-agent-state";
import { useRespondToConfirmation } from "#/hooks/mutation/use-respond-to-confirmation";
import { SecurityRisk } from "#/types/agent-server/core/base/common";

const RISK_ALERT_BY_LEVEL: Partial<
  Record<
    SecurityRisk,
    {
      severity: "high" | "medium";
      titleKey: I18nKey;
      contentKey: I18nKey;
      iconClassName: string;
    }
  >
> = {
  [SecurityRisk.HIGH]: {
    severity: "high",
    titleKey: I18nKey.COMMON$HIGH_RISK,
    contentKey: I18nKey.CHAT_INTERFACE$HIGH_RISK_WARNING,
    iconClassName: "text-danger dark:text-white",
  },
  [SecurityRisk.MEDIUM]: {
    severity: "medium",
    titleKey: I18nKey.COMMON$MEDIUM_RISK,
    contentKey: I18nKey.CHAT_INTERFACE$MEDIUM_RISK_WARNING,
    iconClassName: "text-warning dark:text-white",
  },
};

export function ConversationConfirmationButtons() {
  const submittedEventIds = useEventMessageStore(
    (state) => state.submittedEventIds,
  );
  const addSubmittedEventId = useEventMessageStore(
    (state) => state.addSubmittedEventId,
  );

  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const { mutate: respondToConfirmation } = useRespondToConfirmation();
  const events = useEventStore((state) => state.events);

  const awaitingAction = events
    .slice()
    .reverse()
    .find((ev) => {
      if (ev.source !== "agent") return false;
      return curAgentState === AgentState.AWAITING_USER_CONFIRMATION;
    });

  const hasSubmitted =
    awaitingAction?.id !== undefined &&
    submittedEventIds.includes(awaitingAction.id);

  const handleConfirmation = useCallback(
    (accept: boolean) => {
      if (!awaitingAction || !conversation) {
        return;
      }

      // Mark event as submitted to prevent duplicate submissions
      if (awaitingAction.id) {
        addSubmittedEventId(awaitingAction.id);
      }

      // Call the agent-server API endpoint
      respondToConfirmation({
        conversationId: conversation.id,
        conversationUrl: conversation.conversation_url || "",
        sessionApiKey: conversation.session_api_key,
        accept,
      });
    },
    [awaitingAction, conversation, addSubmittedEventId, respondToConfirmation],
  );

  // Handle keyboard shortcuts
  useEffect(() => {
    // Ignore shortcuts once a response was sent, so a second press cannot
    // submit another (or the opposite) answer before the server update arrives
    if (!awaitingAction || hasSubmitted) {
      return undefined;
    }

    const handleCancelShortcut = (event: KeyboardEvent) => {
      if (
        event.shiftKey &&
        (event.metaKey || event.ctrlKey) &&
        event.key === "Backspace"
      ) {
        event.preventDefault();
        handleConfirmation(false);
      }
    };

    const handleContinueShortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        handleConfirmation(true);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      // Cancel: Shift+Cmd+Backspace (Mac) or Shift+Ctrl+Backspace (Windows/Linux)
      handleCancelShortcut(event);
      // Continue: Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux)
      handleContinueShortcut(event);
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [awaitingAction, hasSubmitted, handleConfirmation]);

  // Only show if agent is waiting for confirmation and we haven't already submitted
  if (
    curAgentState !== AgentState.AWAITING_USER_CONFIRMATION ||
    !awaitingAction ||
    hasSubmitted
  ) {
    return null;
  }

  // Get security risk from the action (only ActionEvent has security_risk)
  const risk = isActionEvent(awaitingAction)
    ? awaitingAction.security_risk
    : SecurityRisk.UNKNOWN;

  const riskAlert = RISK_ALERT_BY_LEVEL[risk];

  return (
    <div className="flex flex-col gap-2 pt-4">
      {riskAlert && (
        <RiskAlert
          content={t(riskAlert.contentKey)}
          icon={
            <WarningIcon
              width={16}
              height={16}
              className={riskAlert.iconClassName}
            />
          }
          severity={riskAlert.severity}
          title={t(riskAlert.titleKey)}
        />
      )}
      <div className="flex justify-between items-center gap-3">
        <p className="min-w-0 text-sm font-normal text-contrast">
          {t(I18nKey.CHAT_INTERFACE$USER_ASK_CONFIRMATION)}
        </p>
        <div className="flex shrink-0 items-center gap-3">
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
