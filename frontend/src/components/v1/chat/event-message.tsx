import React from "react";
import { OpenHandsEvent, MessageEvent, ActionEvent } from "#/types/v1/core";
import { FinishAction } from "#/types/v1/core/base/action";
import {
  isActionEvent,
  isObservationEvent,
  isAgentErrorEvent,
} from "#/types/v1/type-guards";
import { MicroagentStatus } from "#/types/microagent-status";
import { useConfig } from "#/hooks/query/use-config";
// TODO: Implement V1 feedback functionality when API supports V1 event IDs
// import { useFeedbackExists } from "#/hooks/query/use-feedback-exists";
import {
  ErrorEventMessage,
  UserAssistantEventMessage,
  FinishEventMessage,
  GenericEventMessageWrapper,
  ThoughtEventMessage,
} from "./event-message-components";

interface EventMessageProps {
  event: OpenHandsEvent;
  messages: OpenHandsEvent[];
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
  messages,
  isLastMessage,
  microagentStatus,
  microagentConversationId,
  microagentPRUrl,
  actions,
  isInLast10Actions,
}: EventMessageProps) {
  const { data: config } = useConfig();

  // V1 events use string IDs, but useFeedbackExists expects number
  // For now, we'll skip feedback functionality for V1 events
  const feedbackData = { exists: false };
  const isCheckingFeedback = false;

  // Common props for components that need them
  const commonProps = {
    microagentStatus,
    microagentConversationId,
    microagentPRUrl,
    actions,
    isLastMessage,
    isInLast10Actions,
    config,
    isCheckingFeedback,
    feedbackData,
  };

  // Agent error events
  if (isAgentErrorEvent(event)) {
    return <ErrorEventMessage event={event} {...commonProps} />;
  }

  // Finish actions
  if (isActionEvent(event) && event.action.kind === "FinishAction") {
    return (
      <FinishEventMessage
        event={event as ActionEvent<FinishAction>}
        {...commonProps}
      />
    );
  }

  // Action events - render thought + action (will be replaced by thought + observation)
  if (isActionEvent(event)) {
    return (
      <>
        <ThoughtEventMessage event={event} actions={actions} />
        <GenericEventMessageWrapper
          event={event}
          isLastMessage={isLastMessage}
        />
      </>
    );
  }

  // Observation events - find the corresponding action and render thought + observation
  if (isObservationEvent(event)) {
    // Find the action that this observation is responding to
    const correspondingAction = messages.find(
      (msg) => isActionEvent(msg) && msg.id === event.action_id,
    );

    return (
      <>
        {correspondingAction && isActionEvent(correspondingAction) && (
          <ThoughtEventMessage event={correspondingAction} actions={actions} />
        )}
        <GenericEventMessageWrapper
          event={event}
          isLastMessage={isLastMessage}
        />
      </>
    );
  }

  // Message events (user and assistant messages)
  if (!isActionEvent(event) && !isObservationEvent(event)) {
    // This is a MessageEvent
    return (
      <UserAssistantEventMessage
        event={event as MessageEvent}
        {...commonProps}
        isLastMessage={isLastMessage}
      />
    );
  }

  // Generic fallback for all other events
  return (
    <GenericEventMessageWrapper event={event} isLastMessage={isLastMessage} />
  );
}
