import React from "react";
import { ActionEvent } from "#/types/agent-server/core";
import { ChatMessage } from "../../../features/chat/chat-message";
import { CollapsibleThinking } from "./collapsible-thinking";
import {
  getActionThoughtText,
  splitInlineThink,
} from "../event-thought-helpers";

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
  // Some models stream their reasoning inline in the thought instead of via
  // `reasoning_content`, so peel a leading inline reasoning block out before
  // the bubble renders it. `reasoning_content` / `thinking_blocks` are rendered
  // by the caller, so only the inline block is handled here.
  const { reasoning, message } = splitInlineThink(getActionThoughtText(event));

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
