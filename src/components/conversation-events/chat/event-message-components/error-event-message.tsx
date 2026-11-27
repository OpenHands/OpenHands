import React from "react";
import { AgentErrorEvent } from "#/types/agent-server/core";
import { isAgentErrorEvent } from "#/types/agent-server/type-guards";
import { ErrorMessage } from "../../../features/chat/error-message";

interface ErrorEventMessageProps {
  event: AgentErrorEvent;
}

export function ErrorEventMessage({ event }: ErrorEventMessageProps) {
  if (!isAgentErrorEvent(event)) {
    return null;
  }

  return <ErrorMessage errorId={event.id} defaultMessage={event.error} />;
}
