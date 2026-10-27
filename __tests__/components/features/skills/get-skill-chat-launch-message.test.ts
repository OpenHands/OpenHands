import { describe, expect, it } from "vitest";
import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import {
  ADD_SKILL_EXAMPLE_COMMAND,
  ADD_SKILL_SKILL_NAME,
} from "#/constants/skills-docs";
import { getSkillChatLaunchMessage } from "#/components/features/skills/get-skill-chat-launch-message";

describe("getSkillChatLaunchMessage", () => {
  it("returns the full add-skill example command for the add-skill skill", () => {
    expect(getSkillChatLaunchMessage({ name: ADD_SKILL_SKILL_NAME })).toBe(
      ADD_SKILL_EXAMPLE_COMMAND,
    );
  });

  it("points the add-skill example at a skill in the bundled extensions catalog", () => {
    const skillName = ADD_SKILL_EXAMPLE_COMMAND.split("/skills/").at(-1);

    expect(SKILLS_CATALOG.map((entry) => entry.name)).toContain(skillName);
  });

  it("returns a slash-command prefix for other skills", () => {
    expect(getSkillChatLaunchMessage({ name: "agent-builder" })).toBe(
      "/agent-builder ",
    );
  });
});
