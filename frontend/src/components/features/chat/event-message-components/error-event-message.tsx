import React from "react";
import { OpenHandsObservation } from "#/types/core/observations";
import { isErrorObservation } from "#/types/core/guards";
import { ErrorMessage } from "../error-message";

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
    <ErrorMessage
      errorId={event.extras.error_id}
      defaultMessage={event.message}
    />
  );
}
