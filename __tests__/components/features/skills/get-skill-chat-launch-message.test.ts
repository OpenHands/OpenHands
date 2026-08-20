import { describe, expect, it } from "vitest";
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

  it("returns a slash-command prefix for other skills", () => {
    expect(getSkillChatLaunchMessage({ name: "agent-builder" })).toBe(
      "/agent-builder ",
    );
  });
});
