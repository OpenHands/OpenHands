import React from "react";
import { ActionEvent } from "#/types/agent-server/core";
import { ChatMessage } from "../../../features/chat/chat-message";
import { CollapsibleThinking } from "./collapsible-thinking";
import { splitActionNarration } from "../event-thought-helpers";

interface ThoughtEventMessageProps {
  event: ActionEvent;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
  isFromPlanningAgent?: boolean;
}

export function ThoughtEventMessage({
  event,
  actions,
  isFromPlanningAgent = false,
}: ThoughtEventMessageProps) {
  // This component is the single owner of an action's reasoning: explicit
  // `reasoning_content` / `thinking_blocks` merged with a leading inline
  // reasoning block, so the two are never rendered as separate controls.
  const { reasoning, message } = splitActionNarration(event);

  // If there's nothing left to show, don't render anything
  if (!reasoning && !message) {
    return null;
  }

  return (
    <>
      {reasoning && <CollapsibleThinking content={reasoning} />}
      {message && (
        <ChatMessage
          type="agent"
          message={message}
          actions={actions}
          isFromPlanningAgent={isFromPlanningAgent}
          timestamp={event.timestamp}
        />
      )}
    </>
  );
}
