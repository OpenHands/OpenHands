import { MessageEvent } from "#/types/v1/core";
import { BaseEvent } from "#/types/v1/core/base/event";
import { getSkillReadyContent } from "./get-skill-ready-content";

/**
 * Synthetic event type for Skill Ready events.
 * This extends BaseEvent and includes a marker to identify it as a skill ready event.
 */
export interface SkillReadyEvent extends BaseEvent {
  _isSkillReadyEvent: true;
  _skillReadyContent: string;
}

/**
 * Type guard for Skill Ready events.
 */
export const isSkillReadyEvent = (event: unknown): event is SkillReadyEvent =>
  typeof event === "object" &&
  event !== null &&
  "_isSkillReadyEvent" in event &&
  event._isSkillReadyEvent === true;

/**
 * Creates a synthetic "Skill Ready" event from a user MessageEvent.
 * This event appears as originating from the agent and contains formatted
 * information about activated skills and extended content.
 */
export const createSkillReadyEvent = (
  userEvent: MessageEvent,
): SkillReadyEvent => {
  // Support both activated_skills and activated_microagents field names
  const activatedSkills =
    (userEvent as unknown as { activated_skills?: string[] })
      .activated_skills ||
    userEvent.activated_microagents ||
    [];

  const extendedContent = userEvent.extended_content || [];

  // Only create event if we have skills or extended content
  if (activatedSkills.length === 0 && extendedContent.length === 0) {
    throw new Error(
      "Cannot create skill ready event without activated skills or extended content",
    );
  }

  const content = getSkillReadyContent(activatedSkills, extendedContent);

  return {
    id: `${userEvent.id}-skill-ready`,
    timestamp: userEvent.timestamp,
    source: "agent",
    _isSkillReadyEvent: true,
    _skillReadyContent: content,
  };
};
