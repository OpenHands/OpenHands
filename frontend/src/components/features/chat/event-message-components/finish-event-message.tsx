import React from "react";
import { OpenHandsAction } from "#/types/core/actions";
import { isFinishAction } from "#/types/core/guards";
import { ChatMessage } from "../chat-message";
import { getEventContent } from "../event-content-helpers/get-event-content";

interface FinishEventMessageProps {
  event: OpenHandsAction;
}

export function FinishEventMessage({ event }: FinishEventMessageProps) {
  if (!isFinishAction(event)) {
    return null;
  }

  return <ChatMessage type="agent" message={getEventContent(event).details} />;
}
