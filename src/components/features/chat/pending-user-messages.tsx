import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useConversationStore } from "#/stores/conversation-store";
import { useSendMessage } from "#/hooks/use-send-message";
import { createChatMessage } from "#/services/chat-service";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { matchesPendingConversationId } from "#/utils/pending-task-message-link";
import { ImageCarousel } from "#/components/features/images/image-carousel";
import { ChatMessage } from "./chat-message";

/**
 * Renders the queue of locally-tracked user messages that have been submitted
 * but not yet echoed back through the WebSocket. Each message shows a faded
 * "sending" treatment until the server echoes a real `UserMessageEvent`
 * (which removes it via `consumeMatchingPendingMessage`). If the API rejects the
 * send, the message switches to an "error" state with a retry button.
 *
 * The queue is global but each entry is tagged with the conversation id it
 * was enqueued from; this component filters to only entries belonging to the
 * active conversation, so switching conversations never carries pending
 * bubbles over.
 */
export function PendingUserMessages() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useOptionalConversationId();
  const pendingMessages = useOptimisticUserMessageStore(
    (state) => state.pendingMessages,
  );
  const markPendingMessageError = useOptimisticUserMessageStore(
    (state) => state.markPendingMessageError,
  );
  const markPendingMessageSending = useOptimisticUserMessageStore(
    (state) => state.markPendingMessageSending,
  );
  const removePendingMessage = useOptimisticUserMessageStore(
    (state) => state.removePendingMessage,
  );
  const restoreMessageToInputIfEmpty = useConversationStore(
    (state) => state.restoreMessageToInputIfEmpty,
  );
  const { send } = useSendMessage();

  const visibleMessages = React.useMemo(
    () =>
      conversationId
        ? pendingMessages.filter((message) =>
            matchesPendingConversationId(
              conversationId,
              message.conversationId,
            ),
          )
        : [],
    [pendingMessages, conversationId],
  );

  const handleRetry = React.useCallback(
    async (id: string) => {
      const message = useOptimisticUserMessageStore
        .getState()
        .pendingMessages.find((entry) => entry.id === id);
      if (!message) return;

      markPendingMessageSending(id);

      try {
        await send(
          createChatMessage(
            message.text,
            message.imageUrls,
            message.fileUrls,
            message.timestamp,
          ),
        );
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : t(I18nKey.CHAT_INTERFACE$FAILED_TO_SEND_MESSAGE);
        markPendingMessageError(id, errorMessage);
      }
    },
    [send, markPendingMessageError, markPendingMessageSending, t],
  );

  const handleStop = React.useCallback(
    (id: string, text: string) => {
      restoreMessageToInputIfEmpty(text);
      removePendingMessage(id);
    },
    [restoreMessageToInputIfEmpty, removePendingMessage],
  );

  if (visibleMessages.length === 0) {
    return null;
  }

  return (
    <>
      {visibleMessages.map((message) => (
        <ChatMessage
          key={message.id}
          type="user"
          message={message.text}
          pendingStatus={message.status}
          onRetry={
            message.status === "error"
              ? () => handleRetry(message.id)
              : undefined
          }
          onStop={
            message.status === "sending"
              ? () => handleStop(message.id, message.text)
              : undefined
          }
        >
          {message.imageUrls.length > 0 && (
            <ImageCarousel size="small" images={message.imageUrls} />
          )}
        </ChatMessage>
      ))}
    </>
  );
}
