import { OpenHandsEvent, ActionEvent } from "#/types/v1/core";
import { GenericEventMessage } from "../../../features/chat/generic-event-message";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { getObservationResult } from "../event-content-helpers/get-observation-result";
import { isObservationEvent } from "#/types/v1/type-guards";
import {
  SkillReadyEvent,
  isSkillReadyEvent,
} from "../event-content-helpers/create-skill-ready-event";
import { V1ConfirmationButtons } from "#/components/shared/buttons/v1-confirmation-buttons";
import { ObservationResultStatus } from "../../../features/chat/event-content-helpers/get-observation-result";
import { SkillReadyContentList } from "./skill-ready-content-list";

interface GenericEventMessageWrapperProps {
  event: OpenHandsEvent | SkillReadyEvent;
  isLastMessage: boolean;
  correspondingAction?: ActionEvent;
}

export function GenericEventMessageWrapper({
  event,
  isLastMessage,
  correspondingAction,
}: GenericEventMessageWrapperProps) {
  const { title, details } = getEventContent(event, correspondingAction);

  // TaskTrackerObservation has its own rendering
  if (
    !isSkillReadyEvent(event) &&
    isObservationEvent(event) &&
    event.observation.kind === "TaskTrackerObservation"
  ) {
    return <div>{details}</div>;
  }

  // Determine success status
  let success: ObservationResultStatus | undefined;
  if (isSkillReadyEvent(event)) {
    success = "success";
  } else if (isObservationEvent(event)) {
    success = getObservationResult(event);
  }

  // For Skill Ready events with items, render expandable skill list
  const skillReadyDetails =
    isSkillReadyEvent(event) && event._skillReadyItems.length > 0 ? (
      <SkillReadyContentList items={event._skillReadyItems} />
    ) : (
      details
    );

  return (
    <div>
      <GenericEventMessage
        title={title}
        details={skillReadyDetails}
        success={success}
        initiallyExpanded={false}
      />
      {isLastMessage && <V1ConfirmationButtons />}
    </div>
  );
}
