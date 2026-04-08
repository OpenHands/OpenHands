import React from "react";
import { OpenHandsAction } from "#/types/core/actions";
import { isUserMessage, isAssistantMessage } from "#/types/core/guards";
import { ChatMessage } from "../chat-message";
import { ImageCarousel } from "../../images/image-carousel";
import { FileList } from "../../files/file-list";
import { ConfirmationButtons } from "#/components/shared/buttons/confirmation-buttons";
import { LikertScaleWrapper } from "./likert-scale-wrapper";
import { parseMessageFromEvent } from "../event-content-helpers/parse-message-from-event";

interface UserAssistantEventMessageProps {
  event: OpenHandsAction;
  shouldShowConfirmationButtons: boolean;
  isLastMessage: boolean;
  isInLast10Actions: boolean;
  config?: { app_mode?: string } | null;
  isCheckingFeedback: boolean;
  feedbackData: {
    exists: boolean;
    rating?: number;
    reason?: string;
  };
}

export function UserAssistantEventMessage({
  event,
  shouldShowConfirmationButtons,
  isLastMessage,
  isInLast10Actions,
  config,
  isCheckingFeedback,
  feedbackData,
}: UserAssistantEventMessageProps) {
  if (!isUserMessage(event) && !isAssistantMessage(event)) {
    return null;
  }

  const message = parseMessageFromEvent(event);

  return (
    <>
      <ChatMessage type={event.source} message={message}>
        {event.args.image_urls && event.args.image_urls.length > 0 && (
          <ImageCarousel size="small" images={event.args.image_urls} />
        )}
        {event.args.file_urls && event.args.file_urls.length > 0 && (
          <FileList files={event.args.file_urls} />
        )}
        {shouldShowConfirmationButtons && <ConfirmationButtons />}
      </ChatMessage>
      {isAssistantMessage(event) && event.action === "message" && (
        <LikertScaleWrapper
          event={event}
          isLastMessage={isLastMessage}
          isInLast10Actions={isInLast10Actions}
          config={config}
          isCheckingFeedback={isCheckingFeedback}
          feedbackData={feedbackData}
        />
      )}
    </>
  );
}
