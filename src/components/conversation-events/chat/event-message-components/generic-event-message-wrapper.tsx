import {
  OpenHandsEvent,
  ActionEvent,
  ObservationEvent,
} from "#/types/agent-server/core";
import { InvokeSkillObservation } from "#/types/agent-server/core/base/observation";
import { I18nKey } from "#/i18n/declaration";
import { GenericEventMessage } from "../../../features/chat/generic-event-message";
import { getEventContent } from "../event-content-helpers/get-event-content";
import {
  getACPToolCallResult,
  getObservationResult,
  ObservationResultStatus,
} from "../event-content-helpers/get-observation-result";
import {
  isACPToolCallEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import {
  SkillReadyEvent,
  SkillReadyItem,
  isSkillReadyEvent,
} from "../event-content-helpers/create-skill-ready-event";
import { getInvokeSkillItems } from "../event-content-helpers/get-invoke-skill-items";
import { ConversationConfirmationButtons } from "#/components/shared/buttons/conversation-confirmation-buttons";
import { SkillReadyContentList } from "./skill-ready-content-list";
import SkillsIcon from "#/icons/skills.svg?react";

interface GenericEventMessageWrapperProps {
  event: OpenHandsEvent | SkillReadyEvent;
  isLastMessage: boolean;
  correspondingAction?: ActionEvent;
}

/**
 * Resolves the expandable skill-knowledge list shared by Skill Ready events and
 * invoke-skill tool observations. Returns the list items and their header
 * label, or null when the event carries no skill knowledge (so the caller keeps
 * the default details body). Additional skill-knowledge sources slot in as one
 * more branch here.
 */
function getSkillKnowledge(
  event: OpenHandsEvent | SkillReadyEvent,
): { items: SkillReadyItem[]; titleKey: I18nKey } | null {
  if (isSkillReadyEvent(event)) {
    return event._skillReadyItems.length > 0
      ? {
          items: event._skillReadyItems,
          titleKey: I18nKey.SKILLS$TRIGGERED_SKILL_KNOWLEDGE,
        }
      : null;
  }
  if (
    isObservationEvent(event) &&
    event.observation.kind === "InvokeSkillObservation"
  ) {
    const items = getInvokeSkillItems(
      event as ObservationEvent<InvokeSkillObservation>,
    );
    return items.length > 0
      ? { items, titleKey: I18nKey.SKILLS$INVOKED_SKILL_KNOWLEDGE }
      : null;
  }
  return null;
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
  } else if (isACPToolCallEvent(event)) {
    success = getACPToolCallResult(event);
  }

  // Skill Ready events and invoke-skill tool observations both render the
  // expandable skill-knowledge list (with the skills icon); they differ only in
  // the header label.
  const skillKnowledge = getSkillKnowledge(event);
  const bodyDetails = skillKnowledge ? (
    <SkillReadyContentList
      items={skillKnowledge.items}
      titleKey={skillKnowledge.titleKey}
    />
  ) : (
    details
  );

  return (
    <div>
      <GenericEventMessage
        title={title}
        details={bodyDetails}
        success={success}
        initiallyExpanded={false}
        titleIcon={
          skillKnowledge ? (
            <SkillsIcon className="h-4 w-4 stroke-[var(--oh-muted)] flex-shrink-0 mr-2" />
          ) : undefined
        }
      />
      {isLastMessage && <ConversationConfirmationButtons />}
    </div>
  );
}
