import React from "react";
import { OpenHandsEvent, MessageEvent, ActionEvent } from "#/types/v1/core";
import { FinishAction, ThinkAction } from "#/types/v1/core/base/action";
import {
  isActionEvent,
  isObservationEvent,
  isAgentErrorEvent,
  isUserMessageEvent,
  isPlanningFileEditorObservationEvent,
  isHookExecutionEvent,
} from "#/types/v1/type-guards";
import { useConfig } from "#/hooks/query/use-config";
import { useConversationStore } from "#/stores/conversation-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import { ChatMessage } from "../../features/chat/chat-message";
import { PlanPreview } from "../../features/chat/plan-preview";
import {
  ErrorEventMessage,
  UserAssistantEventMessage,
  FinishEventMessage,
  GenericEventMessageWrapper,
  ThoughtEventMessage,
  HookExecutionEventMessage,
} from "./event-message-components";
import { createSkillReadyEvent } from "./event-content-helpers/create-skill-ready-event";
import { shouldShowPlanPreview } from "./hooks/use-plan-preview-events";

interface EventMessageProps {
  event: OpenHandsEvent & { isFromPlanningAgent?: boolean };
  messages: OpenHandsEvent[];
  isLastMessage: boolean;
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
  isLastMessage: boolean;
  isInLast10Actions: boolean;
  config: unknown;
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
    return (
      <UserAssistantEventMessage
        event={messageEvent}
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
  isInLast10Actions,
  planPreviewEventIds,
}: EventMessageProps) {
  const { data: config } = useConfig();
  const { planContent } = useConversationStore();
  const { curAgentState } = useAgentState();

  // Disable Build button while agent is running (streaming)
  const isAgentRunning =
    curAgentState === AgentState.RUNNING ||
    curAgentState === AgentState.LOADING;

  // Read isFromPlanningAgent directly from the event object
  const isFromPlanningAgent = event.isFromPlanningAgent || false;

  // Common props for components that need them
  const commonProps = {
    isLastMessage,
    isInLast10Actions,
    config,
    isFromPlanningAgent,
  };

  // Agent error events
  if (isAgentErrorEvent(event)) {
    return <ErrorEventMessage event={event} {...commonProps} />;
  }

  // Hook execution events
  if (isHookExecutionEvent(event)) {
    return <HookExecutionEventMessage event={event} />;
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

  // ThinkAction - render the thought as a normal chat message (not a collapsible block)
  // The thought content IS the action, so we use event.action.thought directly
  // instead of event.thought (which contains the raw tool call text).
  if (isActionEvent(event) && event.action.kind === "ThinkAction") {
    const thinkAction = event as ActionEvent<ThinkAction>;
    return (
      <ChatMessage
        type="agent"
        message={thinkAction.action.thought}
        isFromPlanningAgent={isFromPlanningAgent}
      />
    );
  }

  // Action events - render thought + action (will be replaced by thought + observation)
  if (isActionEvent(event)) {
    return (
      <>
        <ThoughtEventMessage
          event={event}
          isFromPlanningAgent={isFromPlanningAgent}
        />
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
        // Show shine effect only if this is the last message AND agent is running
        const isStreaming =
          isLastMessage && curAgentState === AgentState.RUNNING;
        return (
          <PlanPreview
            planContent={planContent}
            isStreaming={isStreaming}
            isBuildDisabled={isAgentRunning}
          />
        );
      }
      // Not the designated preview event for this phase - render nothing
      // This prevents duplicate previews within the same phase
      return null;
    }

    // Find the action that this observation is responding to
    const correspondingAction = messages.find(
      (msg) => isActionEvent(msg) && msg.id === event.action_id,
    );

    // Skip ThoughtEventMessage for ThinkAction (thought IS the action)
    const shouldShowThought =
      correspondingAction &&
      isActionEvent(correspondingAction) &&
      correspondingAction.action.kind !== "ThinkAction";

    return (
      <>
        {shouldShowThought && (
          <ThoughtEventMessage
            event={correspondingAction}
            isFromPlanningAgent={isFromPlanningAgent}
          />
        )}
        <GenericEventMessageWrapper
          event={event}
          isLastMessage={isLastMessage}
          correspondingAction={
            correspondingAction && isActionEvent(correspondingAction)
              ? correspondingAction
              : undefined
          }
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
