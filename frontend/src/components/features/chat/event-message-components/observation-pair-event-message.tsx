import React from "react";
import { OpenHandsAction } from "#/types/core/actions";
import { isOpenHandsAction } from "#/types/core/guards";
import { ChatMessage } from "../chat-message";

const hasThoughtProperty = (
  obj: Record<string, unknown>,
): obj is { thought: string } => "thought" in obj && !!obj.thought;

interface ObservationPairEventMessageProps {
  event: OpenHandsAction;
}

export function ObservationPairEventMessage({
  event,
}: ObservationPairEventMessageProps) {
  if (!isOpenHandsAction(event)) {
    return null;
  }

  if (hasThoughtProperty(event.args) && event.action !== "think") {
    return (
      <div>
        <ChatMessage type="agent" message={event.args.thought} />
      </div>
    );
  }

  return null;
}
