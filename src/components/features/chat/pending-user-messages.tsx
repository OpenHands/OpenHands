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
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { useEventStore } from "#/stores/use-event-store";
import { SlashCommandMessages } from "./slash-command-messages";
import { toPendingUserMessageBoundary } from "#/hooks/chat/slash-command-timeline-boundary";

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
export function PendingUserMessages({
  slashCommandOutputScopeId,
}: {
  slashCommandOutputScopeId?: string | null;
}) {
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
  const slashCommandEntries = useSlashCommandOutputStore((state) =>
    slashCommandOutputScopeId
      ? state.entriesByScope[slashCommandOutputScopeId]
      : undefined,
  );
  const resolvePendingMessageBoundary = useSlashCommandOutputStore(
    (state) => state.resolvePendingMessageBoundary,
  );

  const releasePendingBoundary = React.useCallback(
    (id: string) => {
      const pendingMessages =
        useOptimisticUserMessageStore.getState().pendingMessages;
      const pendingIndex = pendingMessages.findIndex(
        (message) => message.id === id,
      );
      const precedingPendingMessage =
        pendingIndex > 0 && conversationId
          ? pendingMessages
              .slice(0, pendingIndex)
              .reverse()
              .find((message) =>
                matchesPendingConversationId(
                  conversationId,
                  message.conversationId,
                ),
              )
          : undefined;
      const lastEventId = useEventStore.getState().events.at(-1)?.id;
      resolvePendingMessageBoundary(
        id,
        precedingPendingMessage
          ? toPendingUserMessageBoundary(precedingPendingMessage.id)
          : lastEventId === undefined || lastEventId === null
            ? null
            : String(lastEventId),
      );
    },
    [conversationId, resolvePendingMessageBoundary],
  );

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
      releasePendingBoundary(id);
      removePendingMessage(id);
    },
    [
      restoreMessageToInputIfEmpty,
      releasePendingBoundary,
      removePendingMessage,
    ],
  );

  const handleDismiss = React.useCallback(
    (id: string) => {
      releasePendingBoundary(id);
      removePendingMessage(id);
    },
    [releasePendingBoundary, removePendingMessage],
  );

  if (visibleMessages.length === 0) {
    return null;
  }

  return (
    <>
      {visibleMessages.map((message) => (
        <React.Fragment key={message.id}>
          <ChatMessage
            type="user"
            message={message.text}
            pendingStatus={message.status}
            onRetry={
              message.status === "error"
                ? () => handleRetry(message.id)
                : undefined
            }
            onDismiss={
              message.status === "error"
                ? () => handleDismiss(message.id)
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
          <SlashCommandMessages
            outputScopeId={slashCommandOutputScopeId}
            outputs={(slashCommandEntries ?? []).filter(
              (entry) =>
                entry.timelineBoundaryEventId ===
                toPendingUserMessageBoundary(message.id),
            )}
          />
        </React.Fragment>
      ))}
    </>
  );
}
