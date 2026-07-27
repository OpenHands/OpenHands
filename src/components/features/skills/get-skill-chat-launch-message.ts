import {
  ADD_SKILL_EXAMPLE_COMMAND,
  ADD_SKILL_SKILL_NAME,
} from "#/constants/skills-docs";
import type { SkillInfo } from "#/types/settings";

export function getSkillChatLaunchMessage(
  skill: Pick<SkillInfo, "name">,
): string {
  if (skill.name === ADD_SKILL_SKILL_NAME) {
    return ADD_SKILL_EXAMPLE_COMMAND;
  }

  return `/${skill.name} `;
}
