import React from "react";
import { OpenHandsAction } from "#/types/core/actions";
import { isFinishAction } from "#/types/core/guards";
import { ChatMessage } from "../chat-message";
import { getEventContent } from "../event-content-helpers/get-event-content";

interface FinishEventMessageProps {
  event: OpenHandsAction;
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

export function FinishEventMessage({
  event,
  isLastMessage,
  isInLast10Actions,
  config,
  isCheckingFeedback,
  feedbackData,
}: FinishEventMessageProps) {
  if (!isFinishAction(event)) {
    return null;
  }

  return (
    <ChatMessage type="agent" message={getEventContent(event).details} />
  );
}
