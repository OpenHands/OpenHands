import { describe, expect, it } from "vitest";
import { getSkillCardDescription } from "#/components/features/skills/get-skill-card-description";
import type { SkillInfo } from "#/types/settings";

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return {
    name: "test",
    type: "knowledge",
    source: null,
    ...overrides,
  };
}

describe("getSkillCardDescription", () => {
  it("prefers the API description field when present", () => {
    expect(
      getSkillCardDescription(
        buildSkill({ description: "From API", content: "ignored body" }),
      ),
    ).toBe("From API");
  });

  it("reads description from YAML frontmatter in content", () => {
    const content = `---
description: SSH helper for remote hosts
name: SSH Microagent
---
# SSH Microagent

Body paragraph that should not be used first.
`;

    expect(getSkillCardDescription(buildSkill({ content }))).toBe(
      "SSH helper for remote hosts",
    );
  });

  it("falls back to the first body paragraph when no description exists", () => {
    const content = `# My Skill

First paragraph for the card.

Second paragraph.`;

    expect(getSkillCardDescription(buildSkill({ content }))).toBe(
      "First paragraph for the card.",
    );
  });

  it("returns an empty string when neither description nor content exists", () => {
    expect(getSkillCardDescription(buildSkill())).toBe("");
  });
});
