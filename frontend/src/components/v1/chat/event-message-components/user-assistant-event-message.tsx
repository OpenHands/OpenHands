import React from "react";
import { MessageEvent } from "#/types/v1/core";
import { ChatMessage } from "../../../features/chat/chat-message";
import { ImageCarousel } from "../../../features/images/image-carousel";
// TODO: Implement file_urls support for V1 messages
// import { FileList } from "../../../features/files/file-list";
import { V1ConfirmationButtons } from "#/components/shared/buttons/v1-confirmation-buttons";
// TODO: Implement V1 LikertScaleWrapper when API supports V1 event IDs
// import { LikertScaleWrapper } from "../../../features/chat/event-message-components/likert-scale-wrapper";
import { parseMessageFromEvent } from "../event-content-helpers/parse-message-from-event";

interface UserAssistantEventMessageProps {
  event: MessageEvent;
  isLastMessage: boolean;
  isFromPlanningAgent: boolean;
}

export function UserAssistantEventMessage({
  event,
  isLastMessage,
  isFromPlanningAgent,
}: UserAssistantEventMessageProps) {
  const message = parseMessageFromEvent(event);

  // Extract image URLs from the message content
  const imageUrls: string[] = [];
  if (Array.isArray(event.llm_message.content)) {
    event.llm_message.content.forEach((content) => {
      if (content.type === "image") {
        imageUrls.push(...content.image_urls);
      }
    });
  }

  return (
    <>
      <ChatMessage
        type={event.source}
        message={message}
        isFromPlanningAgent={isFromPlanningAgent}
      >
        {imageUrls.length > 0 && (
          <ImageCarousel size="small" images={imageUrls} />
        )}
        {/* TODO: Handle file_urls if V1 messages support them */}
        {isLastMessage && <V1ConfirmationButtons />}
      </ChatMessage>
      {/* LikertScaleWrapper expects V0 event types, skip for now */}
    </>
  );
}
