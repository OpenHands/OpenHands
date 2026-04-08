import React from "react";
import { AgentErrorEvent } from "#/types/v1/core";
import { isAgentErrorEvent } from "#/types/v1/type-guards";
import { ErrorMessage } from "../../../features/chat/error-message";

interface ErrorEventMessageProps {
  event: AgentErrorEvent;
}

export function ErrorEventMessage({ event }: ErrorEventMessageProps) {
  if (!isAgentErrorEvent(event)) {
    return null;
  }

  return (
    <div>
      <ErrorMessage
        // V1 doesn't have error_id, use event.id instead
        errorId={event.id}
        defaultMessage={event.error}
      />
      {/* LikertScaleWrapper expects V0 event types, skip for now */}
    </div>
  );
}
