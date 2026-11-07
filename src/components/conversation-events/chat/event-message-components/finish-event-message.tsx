import { ActionEvent } from "#/types/agent-server/core";
import { FinishAction } from "#/types/agent-server/core/base/action";
import { ChatMessage } from "../../../features/chat/chat-message";
import {
  MessageReactions,
  useHasMessageReactions,
} from "../../../features/chat/message-reactions";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { CriticResultDisplay } from "./critic-result-display";

interface FinishEventMessageProps {
  event: ActionEvent<FinishAction>;
  isFromPlanningAgent?: boolean;
}

export function FinishEventMessage({
  event,
  isFromPlanningAgent = false,
}: FinishEventMessageProps) {
  const showReactions = useHasMessageReactions(event.id, "agent");
  const eventContent = getEventContent(event);
  const message =
    typeof eventContent.details === "string"
      ? eventContent.details
      : String(eventContent.details);

  return (
    <>
      <ChatMessage
        type="agent"
        message={message}
        isFromPlanningAgent={isFromPlanningAgent}
        timestamp={event.timestamp}
      >
        {showReactions && (
          <MessageReactions eventId={event.id} author="agent" />
        )}
      </ChatMessage>
      {event.critic_result != null && (
        <CriticResultDisplay criticResult={event.critic_result} />
      )}
    </>
  );
}
