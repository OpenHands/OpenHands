import { ActionEvent } from "#/types/agent-server/core";
import { FinishAction } from "#/types/agent-server/core/base/action";
import { ChatMessage } from "../../../features/chat/chat-message";
import {
  MessageReactions,
  useMessageReactions,
} from "../../../features/chat/message-reactions";
import {
  MessageEditedMark,
  useMessageEdit,
} from "../../../features/chat/message-edits";
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
  const reactions = useMessageReactions(event.id, "agent", {
    isFromPlanningAgent,
  });
  const editedText = useMessageEdit(event.id);
  const eventContent = getEventContent(event);
  const message =
    editedText ??
    (typeof eventContent.details === "string"
      ? eventContent.details
      : String(eventContent.details));

  return (
    <>
      <ChatMessage
        type="agent"
        message={message}
        isFromPlanningAgent={isFromPlanningAgent}
        timestamp={event.timestamp}
      >
        {editedText !== undefined && <MessageEditedMark />}
        {reactions.visible && <MessageReactions state={reactions} />}
      </ChatMessage>
      {event.critic_result != null && (
        <CriticResultDisplay criticResult={event.critic_result} />
      )}
    </>
  );
}
