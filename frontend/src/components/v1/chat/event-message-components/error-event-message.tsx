import React from "react";
import { AgentErrorEvent } from "#/types/v1/core";
import { isAgentErrorEvent } from "#/types/v1/type-guards";
import { ErrorMessage } from "../../../features/chat/error-message";
import { MicroagentStatusWrapper } from "../../../features/chat/event-message-components/microagent-status-wrapper";
// TODO: Implement V1 LikertScaleWrapper when API supports V1 event IDs
// import { LikertScaleWrapper } from "../../../features/chat/event-message-components/likert-scale-wrapper";
import { MicroagentStatus } from "#/types/microagent-status";

interface ErrorEventMessageProps {
  event: AgentErrorEvent;
  microagentStatus?: MicroagentStatus | null;
  microagentConversationId?: string;
  microagentPRUrl?: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
}

export function ErrorEventMessage({
  event,
  microagentStatus,
  microagentConversationId,
  microagentPRUrl,
  actions,
}: ErrorEventMessageProps) {
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
      <MicroagentStatusWrapper
        microagentStatus={microagentStatus}
        microagentConversationId={microagentConversationId}
        microagentPRUrl={microagentPRUrl}
        actions={actions}
      />
      {/* LikertScaleWrapper expects V0 event types, skip for now */}
    </div>
  );
}
