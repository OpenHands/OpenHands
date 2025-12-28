import React from "react";
import { ActionEvent } from "#/types/v1/core";
import { isActionEvent } from "#/types/v1/type-guards";
import { ChatMessage } from "../../../features/chat/chat-message";
import { MicroagentStatusWrapper } from "../../../features/chat/event-message-components/microagent-status-wrapper";
import { MicroagentStatus } from "#/types/microagent-status";

interface ObservationPairEventMessageProps {
  event: ActionEvent;
  microagentStatus?: MicroagentStatus | null;
  microagentConversationId?: string;
  microagentPRUrl?: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
}

export function ObservationPairEventMessage({
  event,
  microagentStatus,
  microagentConversationId,
  microagentPRUrl,
  actions,
}: ObservationPairEventMessageProps) {
  if (!isActionEvent(event)) {
    return null;
  }

  // Check if there's thought content to display
  const thoughtContent = event.thought
    .filter((t) => t.type === "text")
    .map((t) => t.text)
    .join("\n");

  // Defensive check: ensure action exists and has kind property
  if (
    thoughtContent &&
    event.action?.kind &&
    event.action.kind !== "ThinkAction"
  ) {
    return (
      <div>
        <ChatMessage type="agent" message={thoughtContent} actions={actions} />
        <MicroagentStatusWrapper
          microagentStatus={microagentStatus}
          microagentConversationId={microagentConversationId}
          microagentPRUrl={microagentPRUrl}
          actions={actions}
        />
      </div>
    );
  }

  return (
    <MicroagentStatusWrapper
      microagentStatus={microagentStatus}
      microagentConversationId={microagentConversationId}
      microagentPRUrl={microagentPRUrl}
      actions={actions}
    />
  );
}
