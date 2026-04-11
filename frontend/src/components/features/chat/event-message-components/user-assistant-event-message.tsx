import { OpenHandsAction } from "#/types/core/actions";
import { isUserMessage, isAssistantMessage } from "#/types/core/guards";
import { ChatMessage } from "../chat-message";
import { ImageCarousel } from "../../images/image-carousel";
import { FileList } from "../../files/file-list";
import { ConfirmationButtons } from "#/components/shared/buttons/confirmation-buttons";
import { parseMessageFromEvent } from "../event-content-helpers/parse-message-from-event";

interface UserAssistantEventMessageProps {
  event: OpenHandsAction;
  shouldShowConfirmationButtons: boolean;
}

export function UserAssistantEventMessage({
  event,
  shouldShowConfirmationButtons,
}: UserAssistantEventMessageProps) {
  if (!isUserMessage(event) && !isAssistantMessage(event)) {
    return null;
  }

  const message = parseMessageFromEvent(event);

  return (
    <ChatMessage type={event.source} message={message}>
      {event.args.image_urls && event.args.image_urls.length > 0 && (
        <ImageCarousel size="small" images={event.args.image_urls} />
      )}
      {event.args.file_urls && event.args.file_urls.length > 0 && (
        <FileList files={event.args.file_urls} />
      )}
      {shouldShowConfirmationButtons && <ConfirmationButtons />}
    </ChatMessage>
  );
}
