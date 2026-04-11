import React from "react";
import { ActionEvent } from "#/types/v1/core";
import { isActionEvent } from "#/types/v1/type-guards";
import { ChatMessage } from "../../../features/chat/chat-message";

interface ObservationPairEventMessageProps {
  event: ActionEvent;
}

export function ObservationPairEventMessage({
  event,
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
        <ChatMessage type="agent" message={thoughtContent} />
      </div>
    );
  }

  return null;
}
