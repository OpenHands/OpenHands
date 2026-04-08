import React from "react";
import { OpenHandsAction } from "#/types/core/actions";
import {
  isUserMessage,
  isErrorObservation,
  isAssistantMessage,
  isOpenHandsAction,
  isFinishAction,
  isRejectObservation,
  isMcpObservation,
  isTaskTrackingObservation,
} from "#/types/core/guards";
import { OpenHandsObservation } from "#/types/core/observations";
import { useConfig } from "#/hooks/query/use-config";
import { useFeedbackExists } from "#/hooks/query/use-feedback-exists";
import {
  ErrorEventMessage,
  UserAssistantEventMessage,
  FinishEventMessage,
  RejectEventMessage,
  McpEventMessage,
  TaskTrackingEventMessage,
  ObservationPairEventMessage,
  GenericEventMessageWrapper,
} from "./event-message-components";

interface EventMessageProps {
  event: OpenHandsAction | OpenHandsObservation;
  hasObservationPair: boolean;
  isAwaitingUserConfirmation: boolean;
  isLastMessage: boolean;
  isInLast10Actions: boolean;
}

/* eslint-disable react/jsx-props-no-spreading */
export function EventMessage({
  event,
  hasObservationPair,
  isAwaitingUserConfirmation,
  isLastMessage,
  isInLast10Actions,
}: EventMessageProps) {
  const shouldShowConfirmationButtons =
    isLastMessage && event.source === "agent" && isAwaitingUserConfirmation;

  const { data: config } = useConfig();

  const {
    data: feedbackData = { exists: false },
    isLoading: isCheckingFeedback,
  } = useFeedbackExists(event.id);

  // Common props for components that need them
  const commonProps = {
    isLastMessage,
    isInLast10Actions,
    config,
    isCheckingFeedback,
    feedbackData,
  };

  // Error observations
  if (isErrorObservation(event)) {
    return <ErrorEventMessage event={event} {...commonProps} />;
  }

  // Observation pairs with OpenHands actions
  if (hasObservationPair && isOpenHandsAction(event)) {
    return <ObservationPairEventMessage event={event} />;
  }

  // Finish actions
  if (isFinishAction(event)) {
    return <FinishEventMessage event={event} {...commonProps} />;
  }

  // User and assistant messages
  if (isUserMessage(event) || isAssistantMessage(event)) {
    return (
      <UserAssistantEventMessage
        event={event}
        shouldShowConfirmationButtons={shouldShowConfirmationButtons}
        {...commonProps}
      />
    );
  }

  // Reject observations
  if (isRejectObservation(event)) {
    return <RejectEventMessage event={event} />;
  }

  // MCP observations
  if (isMcpObservation(event)) {
    return (
      <McpEventMessage
        event={event}
        shouldShowConfirmationButtons={shouldShowConfirmationButtons}
      />
    );
  }

  // Task tracking observations
  if (isTaskTrackingObservation(event)) {
    return (
      <TaskTrackingEventMessage
        event={event}
        shouldShowConfirmationButtons={shouldShowConfirmationButtons}
      />
    );
  }

  // Generic fallback
  return (
    <GenericEventMessageWrapper
      event={event}
      shouldShowConfirmationButtons={shouldShowConfirmationButtons}
    />
  );
}
