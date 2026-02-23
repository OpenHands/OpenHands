import React from "react";
import { ActionEvent } from "#/types/v1/core";
import { FinishAction } from "#/types/v1/core/base/action";
import { ChatMessage } from "../../../features/chat/chat-message";
import { MicroagentStatusWrapper } from "../../../features/chat/event-message-components/microagent-status-wrapper";
// TODO: Implement V1 LikertScaleWrapper when API supports V1 event IDs
// import { LikertScaleWrapper } from "../../../features/chat/event-message-components/likert-scale-wrapper";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { MicroagentStatus } from "#/types/microagent-status";

interface FinishEventMessageProps {
  event: ActionEvent<FinishAction>;
  microagentStatus?: MicroagentStatus | null;
  microagentConversationId?: string;
  microagentPRUrl?: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
  isFromPlanningAgent?: boolean;
}

export function FinishEventMessage({
  event,
  microagentStatus,
  microagentConversationId,
  microagentPRUrl,
  actions,
  isFromPlanningAgent = false,
}: FinishEventMessageProps) {
  const eventContent = getEventContent(event);
  // For FinishAction, details is always a string (getActionContent returns string)
  const message =
    typeof eventContent.details === "string"
      ? eventContent.details
      : String(eventContent.details);

  return (
    <>
      <ChatMessage
        type="agent"
        message={message}
        actions={actions}
        isFromPlanningAgent={isFromPlanningAgent}
      />
      <MicroagentStatusWrapper
        microagentStatus={microagentStatus}
        microagentConversationId={microagentConversationId}
        microagentPRUrl={microagentPRUrl}
        actions={actions}
      />
      {/* LikertScaleWrapper expects V0 event types, skip for now */}
    </>
  );
}
