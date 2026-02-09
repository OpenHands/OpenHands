import React from "react";
import { OpenHandsObservation } from "#/types/core/observations";
import { isErrorObservation } from "#/types/core/guards";
import { ErrorMessage } from "../error-message";
import { MicroagentStatusWrapper } from "./microagent-status-wrapper";
import { LikertScaleWrapper } from "./likert-scale-wrapper";
import { MicroagentStatus } from "#/types/microagent-status";

interface ErrorEventMessageProps {
  event: OpenHandsObservation;
  microagentStatus?: MicroagentStatus | null;
  microagentConversationId?: string;
  microagentPRUrl?: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
  isLastMessage: boolean;
  isInLast10Actions: boolean;
  config?: { app_mode?: string } | null;
  isCheckingFeedback: boolean;
  feedbackData: {
    exists: boolean;
    rating?: number;
    reason?: string;
  };
}

export function ErrorEventMessage({
  event,
  microagentStatus,
  microagentConversationId,
  microagentPRUrl,
  actions,
  isLastMessage,
  isInLast10Actions,
  config,
  isCheckingFeedback,
  feedbackData,
}: ErrorEventMessageProps) {
  if (!isErrorObservation(event)) {
    return null;
  }

  return (
    <div>
      <ErrorMessage
        errorId={event.extras.error_id}
        defaultMessage={event.message}
      />
      <MicroagentStatusWrapper
        microagentStatus={microagentStatus}
        microagentConversationId={microagentConversationId}
        microagentPRUrl={microagentPRUrl}
        actions={actions}
      />
      <LikertScaleWrapper
        event={event}
        isLastMessage={isLastMessage}
        isInLast10Actions={isInLast10Actions}
        config={config}
        isCheckingFeedback={isCheckingFeedback}
        feedbackData={feedbackData}
      />
    </div>
  );
}
