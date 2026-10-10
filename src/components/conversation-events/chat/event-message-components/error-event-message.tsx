import React from "react";
import {
  AgentErrorEvent,
  ConversationErrorEvent,
} from "#/types/agent-server/core";
import {
  isAgentErrorEvent,
  isConversationErrorEvent,
} from "#/types/agent-server/type-guards";
import { ErrorMessage } from "../../../features/chat/error-message";

interface ErrorEventMessageProps {
  event: AgentErrorEvent | ConversationErrorEvent;
}

export function ErrorEventMessage({ event }: ErrorEventMessageProps) {
  // @spec CE-001 — Conversation errors remain readable in chat history
  if (isConversationErrorEvent(event)) {
    return <ErrorMessage errorId={event.code} defaultMessage={event.detail} />;
  }

  if (!isAgentErrorEvent(event)) {
    return null;
  }

  return <ErrorMessage errorId={event.id} defaultMessage={event.error} />;
}
