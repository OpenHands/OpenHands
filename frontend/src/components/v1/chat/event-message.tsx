import React from "react";
import { OpenHandsEvent, MessageEvent, ActionEvent } from "#/types/v1/core";
import { FinishAction } from "#/types/v1/core/base/action";
import {
  isActionEvent,
  isObservationEvent,
  isAgentErrorEvent,
  isUserMessageEvent,
  isPlanningFileEditorObservationEvent,
} from "#/types/v1/type-guards";
import { MicroagentStatus } from "#/types/microagent-status";
import { useConfig } from "#/hooks/query/use-config";
import { useConversationStore } from "#/stores/conversation-store";
// TODO: Implement V1 feedback functionality when API supports V1 event IDs
// import { useFeedbackExists } from "#/hooks/query/use-feedback-exists";
import {
  ErrorEventMessage,
  UserAssistantEventMessage,
  FinishEventMessage,
  GenericEventMessageWrapper,
  ThoughtEventMessage,
} from "./event-message-components";
import { createSkillReadyEvent } from "./event-content-helpers/create-skill-ready-event";
import { PlanPreview } from "../../features/chat/plan-preview";
import { shouldShowPlanPreview } from "./hooks/use-plan-preview-events";

interface EventMessageProps {
  event: OpenHandsEvent & { isFromPlanningAgent?: boolean };
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
  /** Set of event IDs that should render PlanPreview (one per user message phase) */
  planPreviewEventIds?: Set<string>;
}

/**
 * Extracts activated skills from a MessageEvent, supporting both
 * activated_skills and activated_microagents field names.
 */
const getActivatedSkills = (event: MessageEvent): string[] =>
  (event as unknown as { activated_skills?: string[] }).activated_skills ||
  event.activated_microagents ||
  [];

/**
 * Checks if extended content contains valid text content.
 */
const hasValidExtendedContent = (
  extendedContent: MessageEvent["extended_content"],
): boolean => {
  if (!extendedContent || extendedContent.length === 0) {
    return false;
  }

  return extendedContent.some(
    (content) => content.type === "text" && content.text.trim().length > 0,
  );
};

/**
 * Determines if a Skill Ready event should be displayed for the given message event.
 */
const shouldShowSkillReadyEvent = (messageEvent: MessageEvent): boolean => {
  const activatedSkills = getActivatedSkills(messageEvent);
  const hasActivatedSkills = activatedSkills.length > 0;
  const hasExtendedContent = hasValidExtendedContent(
    messageEvent.extended_content,
  );

  return hasActivatedSkills && hasExtendedContent;
};

interface CommonProps {
  microagentStatus?: MicroagentStatus | null;
  microagentConversationId?: string;
  microagentPRUrl?: string;
  actions?: Array<{
    icon: React.ReactNode;
    onClick: () => void;
    tooltip?: string;
  }>;
  isLastMessage: boolean;
  isInLast10Actions: boolean;
  config: unknown;
  isCheckingFeedback: boolean;
  feedbackData: { exists: boolean };
  isFromPlanningAgent: boolean;
}

/**
 * Renders a user message with its corresponding Skill Ready event.
 */
const renderUserMessageWithSkillReady = (
  messageEvent: MessageEvent,
  commonProps: CommonProps,
  isLastMessage: boolean,
): React.ReactElement => {
  try {
    const skillReadyEvent = createSkillReadyEvent(messageEvent);
    return (
      <>
        <UserAssistantEventMessage
          event={messageEvent}
          microagentStatus={commonProps.microagentStatus}
          microagentConversationId={commonProps.microagentConversationId}
          microagentPRUrl={commonProps.microagentPRUrl}
          actions={commonProps.actions}
          isLastMessage={false}
          isFromPlanningAgent={commonProps.isFromPlanningAgent}
        />
        <GenericEventMessageWrapper
          event={skillReadyEvent}
          isLastMessage={isLastMessage}
        />
      </>
    );
  } catch (error) {
    // If skill ready event creation fails, just render the user message
    // Failed to create skill ready event, fallback to user message
    return (
      <UserAssistantEventMessage
        event={messageEvent}
        microagentStatus={commonProps.microagentStatus}
        microagentConversationId={commonProps.microagentConversationId}
        microagentPRUrl={commonProps.microagentPRUrl}
        actions={commonProps.actions}
        isLastMessage={isLastMessage}
        isFromPlanningAgent={commonProps.isFromPlanningAgent}
      />
    );
  }
};

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
  planPreviewEventIds,
}: EventMessageProps) {
  const { data: config } = useConfig();
  const { planContent } = useConversationStore();

  // V1 events use string IDs, but useFeedbackExists expects number
  // For now, we'll skip feedback functionality for V1 events
  const feedbackData = { exists: false };
  const isCheckingFeedback = false;

  // Read isFromPlanningAgent directly from the event object
  const isFromPlanningAgent = event.isFromPlanningAgent || false;

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
    isFromPlanningAgent,
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
    // Handle PlanningFileEditorObservation specially
    if (isPlanningFileEditorObservationEvent(event)) {
      // Only show PlanPreview if this event is marked as the one to display
      // (last PlanningFileEditorObservation in its phase)
      if (
        planPreviewEventIds &&
        shouldShowPlanPreview(event.id, planPreviewEventIds)
      ) {
        return <PlanPreview planContent={planContent} />;
      }
      // Not the designated preview event for this phase - render nothing
      // This prevents duplicate previews within the same phase
      return null;
    }

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
    const messageEvent = event as MessageEvent;

    // Check if this is a user message that should display a Skill Ready event
    if (isUserMessageEvent(event) && shouldShowSkillReadyEvent(messageEvent)) {
      return renderUserMessageWithSkillReady(
        messageEvent,
        commonProps,
        isLastMessage,
      );
    }

    // Render normal message event (user or assistant)
    return (
      <UserAssistantEventMessage
        event={messageEvent}
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
