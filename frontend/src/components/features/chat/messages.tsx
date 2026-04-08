import React from "react";
import { OpenHandsAction } from "#/types/core/actions";
import { OpenHandsObservation } from "#/types/core/observations";
import { isOpenHandsAction, isOpenHandsObservation } from "#/types/core/guards";
import { EventMessage } from "./event-message";
import { ChatMessage } from "./chat-message";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";

interface MessagesProps {
  messages: (OpenHandsAction | OpenHandsObservation)[];
  isAwaitingUserConfirmation: boolean;
}

export const Messages: React.FC<MessagesProps> = React.memo(
  ({ messages, isAwaitingUserConfirmation }) => {
    const { getOptimisticUserMessage } = useOptimisticUserMessageStore();
    const optimisticUserMessage = getOptimisticUserMessage();

    const actionHasObservationPair = React.useCallback(
      (event: OpenHandsAction | OpenHandsObservation): boolean => {
        if (isOpenHandsAction(event)) {
          return !!messages.some(
            (msg) => isOpenHandsObservation(msg) && msg.cause === event.id,
          );
        }

        return false;
      },
      [messages],
    );

    return (
      <>
        {messages.map((message, index) => (
          <EventMessage
            key={index}
            event={message}
            hasObservationPair={actionHasObservationPair(message)}
            isAwaitingUserConfirmation={isAwaitingUserConfirmation}
            isLastMessage={messages.length - 1 === index}
            isInLast10Actions={messages.length - 1 - index < 10}
          />
        ))}

        {optimisticUserMessage && (
          <ChatMessage type="user" message={optimisticUserMessage} />
        )}
      </>
    );
  },
  (prevProps, nextProps) => {
    // Prevent re-renders if messages are the same length
    if (prevProps.messages.length !== nextProps.messages.length) {
      return false;
    }

    return true;
  },
);

Messages.displayName = "Messages";
