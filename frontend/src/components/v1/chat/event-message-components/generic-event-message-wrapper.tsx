import { OpenHandsEvent } from "#/types/v1/core";
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

interface GenericEventMessageWrapperProps {
  event: OpenHandsEvent | SkillReadyEvent;
  isLastMessage: boolean;
}

export function GenericEventMessageWrapper({
  event,
  isLastMessage,
}: GenericEventMessageWrapperProps) {
  const { title, details } = getEventContent(event);

  // SkillReadyEvent is not an observation event, so skip the observation checks
  if (!isSkillReadyEvent(event)) {
    if (isObservationEvent(event)) {
      if (event.observation.kind === "TaskTrackerObservation") {
        return <div>{details}</div>;
      }
    }
  }

  // Determine success status
  let success: ObservationResultStatus | undefined;
  if (isSkillReadyEvent(event)) {
    // Skill Ready events should show success indicator, same as v0 recall observations
    success = "success";
  } else if (isObservationEvent(event)) {
    success = getObservationResult(event);
  }

  return (
    <div>
      <GenericEventMessage
        title={title}
        details={details}
        success={success}
        initiallyExpanded={false}
      />
      {isLastMessage && <V1ConfirmationButtons />}
    </div>
  );
}
