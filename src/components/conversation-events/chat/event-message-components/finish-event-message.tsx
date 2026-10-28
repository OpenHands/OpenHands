import { useTranslation } from "react-i18next";
import { ActionEvent } from "#/types/agent-server/core";
import { FinishAction } from "#/types/agent-server/core/base/action";
import { ChatMessage } from "../../../features/chat/chat-message";
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
  const { i18n } = useTranslation("openhands");
  const eventContent = getEventContent(event, undefined, i18n);
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
      />
      {event.critic_result != null && (
        <CriticResultDisplay criticResult={event.critic_result} />
      )}
    </>
  );
}
