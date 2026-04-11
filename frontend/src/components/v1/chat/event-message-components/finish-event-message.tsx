import { ActionEvent } from "#/types/v1/core";
import { FinishAction } from "#/types/v1/core/base/action";
import { ChatMessage } from "../../../features/chat/chat-message";
import { getEventContent } from "../event-content-helpers/get-event-content";

interface FinishEventMessageProps {
  event: ActionEvent<FinishAction>;
  isFromPlanningAgent?: boolean;
}

export function FinishEventMessage({
  event,
  isFromPlanningAgent = false,
}: FinishEventMessageProps) {
  const eventContent = getEventContent(event);
  const message =
    typeof eventContent.details === "string"
      ? eventContent.details
      : String(eventContent.details);

  return (
    <ChatMessage
      type="agent"
      message={message}
      isFromPlanningAgent={isFromPlanningAgent}
    />
  );
}
