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
import { MicroagentStatus } from "#/types/microagent-status";
import { useConfig } from "#/hooks/query/use-config";
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
  microagentStatus?: MicroagentStatus | null;
  microagentConversationId?: string;
  microagentPRUrl?: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
  isInLast10Actions: boolean;
}

/* eslint-disable react/jsx-props-no-spreading */
export function EventMessage({
  event,
  hasObservationPair,
  isAwaitingUserConfirmation,
  isLastMessage,
  microagentStatus,
  microagentConversationId,
  microagentPRUrl,
  actions,
  isInLast10Actions,
}: EventMessageProps) {
  const shouldShowConfirmationButtons =
    isLastMessage && event.source === "agent" && isAwaitingUserConfirmation;

  const { data: config } = useConfig();

  // Common props for components that need them
  const commonProps = {
    microagentStatus,
    microagentConversationId,
    microagentPRUrl,
    actions,
    isLastMessage,
    isInLast10Actions,
    config,
  };

  // Error observations
  if (isErrorObservation(event)) {
    return <ErrorEventMessage event={event} {...commonProps} />;
  }

  // Observation pairs with OpenHands actions
  if (hasObservationPair && isOpenHandsAction(event)) {
    return (
      <ObservationPairEventMessage
        event={event}
        microagentStatus={microagentStatus}
        microagentConversationId={microagentConversationId}
        microagentPRUrl={microagentPRUrl}
        actions={actions}
      />
    );
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
