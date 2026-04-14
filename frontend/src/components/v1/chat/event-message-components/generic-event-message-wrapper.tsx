import React from "react";
import { ActionEvent, OpenHandsEvent } from "#/types/v1/core";
import { GenericEventMessage } from "../../../features/chat/generic-event-message";
import { getEventContent } from "../event-content-helpers/get-event-content";
import { getActionContent } from "../event-content-helpers/get-action-content";
import { getObservationResult } from "../event-content-helpers/get-observation-result";
import { isObservationEvent } from "#/types/v1/type-guards";
import {
  SkillReadyEvent,
  isSkillReadyEvent,
} from "../event-content-helpers/create-skill-ready-event";
import { V1ConfirmationButtons } from "#/components/shared/buttons/v1-confirmation-buttons";
import { ObservationResultStatus } from "../../../features/chat/event-content-helpers/get-observation-result";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";

interface GenericEventMessageWrapperProps {
  event: OpenHandsEvent | SkillReadyEvent;
  isLastMessage: boolean;
  /**
   * Optional action event that produced this observation. When provided,
   * the action's input/arguments are prepended to the observation details
   * so users can see both the call inputs and the result in one block.
   */
  actionEvent?: ActionEvent;
}

export function GenericEventMessageWrapper({
  event,
  isLastMessage,
  actionEvent,
}: GenericEventMessageWrapperProps) {
  const { title, details } = getEventContent(event);

  // If this wrapper is rendering an observation and we have the corresponding
  // action, splice the action's input content (e.g. MCP tool arguments) into
  // the observation details so users see it inside the same collapsible block.
  // The action content is inserted *after* the observation header (e.g.
  // "**Tool:** name") but *before* the Result/Error/Output section so the
  // tool description still appears first.
  let mergedDetails: string | React.ReactNode = details;
  if (actionEvent && typeof details === "string") {
    const actionContent = getActionContent(actionEvent);
    if (actionContent) {
      const resultMatch = details.match(/\n*\*\*(?:Result|Error|Output):\*\*/);
      if (resultMatch?.index !== undefined) {
        const head = details.slice(0, resultMatch.index).trimEnd();
        const tail = details.slice(resultMatch.index).trimStart();
        mergedDetails = `${head}\n\n${actionContent}\n\n${tail}`;
      } else {
        mergedDetails = `${details}\n\n${actionContent}`;
      }
    }
  }

  // SkillReadyEvent is not an observation event, so skip the observation checks
  if (!isSkillReadyEvent(event)) {
    if (isObservationEvent(event)) {
      if (event.observation.kind === "TaskTrackerObservation") {
        return <div>{details}</div>;
      }
      if (event.observation.kind === "FinishObservation") {
        return (
          <MarkdownRenderer includeStandard includeHeadings>
            {details as string}
          </MarkdownRenderer>
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
        details={mergedDetails}
        success={success}
        initiallyExpanded={false}
      />
      {isLastMessage && <V1ConfirmationButtons />}
    </div>
  );
}
