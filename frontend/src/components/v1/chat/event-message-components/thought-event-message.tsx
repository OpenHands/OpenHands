import React from "react";
import { ActionEvent } from "#/types/v1/core";
import { ChatMessage } from "../../../features/chat/chat-message";
import { ReasoningContent } from "./reasoning-content";

interface ThoughtEventMessageProps {
  event: ActionEvent;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
}

export function ThoughtEventMessage({
  event,
  actions,
}: ThoughtEventMessageProps) {
  // Extract thought content from the action event
  const thoughtContent = event.thought
    .filter((t) => t.type === "text")
    .map((t) => t.text)
    .join("\n");

  // Check if there's reasoning content to display
  const hasReasoningContent =
    event.reasoning_content && event.reasoning_content.trim().length > 0;
  const hasThinkingBlocks =
    event.thinking_blocks && event.thinking_blocks.length > 0;

  if (!thoughtContent && !hasReasoningContent && !hasThinkingBlocks) {
    return null;
  }

  if (!thoughtContent && (hasReasoningContent || hasThinkingBlocks)) {
    return <ReasoningContent event={event} />;
  }
  return (
    <div>
      <ChatMessage type="agent" message={thoughtContent} actions={actions} />
      <ReasoningContent event={event} />
    </div>
  );
}
