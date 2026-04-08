import React from "react";
import { OpenHandsObservation } from "#/types/core/observations";
import { isErrorObservation } from "#/types/core/guards";
import { ErrorMessage } from "../error-message";
import { LikertScaleWrapper } from "./likert-scale-wrapper";

interface ErrorEventMessageProps {
  event: OpenHandsObservation;
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
