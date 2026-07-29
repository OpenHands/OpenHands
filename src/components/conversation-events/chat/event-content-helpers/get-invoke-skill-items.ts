import { ObservationEvent } from "#/types/agent-server/core";
import { InvokeSkillObservation } from "#/types/agent-server/core/base/observation";
import { SkillReadyItem } from "./get-skill-ready-content";

/**
 * Maps an InvokeSkillObservation to the SkillReadyItem shape so invoke-skill
 * tool calls can reuse the Skill Ready expandable list. A single skill is
 * invoked per observation, so this yields at most one item. Returns an empty
 * array when there is neither a skill name nor any text content to show, which
 * tells the caller to fall back to the default markdown body.
 */
export const getInvokeSkillItems = (
  event: ObservationEvent<InvokeSkillObservation>,
): SkillReadyItem[] => {
  const { observation } = event;

  const content = observation.content
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("\n")
    .trim();

  if (!observation.skill_name && !content) {
    return [];
  }

  return [{ name: observation.skill_name, content }];
};
