import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
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
import { code } from "#/components/features/markdown/code";
import { ul, ol } from "#/components/features/markdown/list";
import { paragraph } from "#/components/features/markdown/paragraph";
import { anchor } from "#/components/features/markdown/anchor";
import {
  h1,
  h2,
  h3,
  h4,
  h5,
  h6,
} from "#/components/features/markdown/headings";

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
      if (event.observation.kind === "FinishObservation") {
        return (
          <Markdown
            components={{
              code,
              ul,
              ol,
              a: anchor,
              p: paragraph,
              h1,
              h2,
              h3,
              h4,
              h5,
              h6,
            }}
            remarkPlugins={[remarkGfm, remarkBreaks]}
          >
            {details as string}
          </Markdown>
        );
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
