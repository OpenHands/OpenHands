import { describe, expect, it } from "vitest";
import type { SkillInfo } from "#/types/settings";
import { getSkillKey } from "#/utils/skill-key";

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return {
    name: "skill-creator",
    type: "knowledge",
    source: null,
    ...overrides,
  };
}

describe("getSkillKey", () => {
  it("distinguishes a local skill from the bundled entry of the same name", () => {
    const local = buildSkill({
      source: "/Users/test/.agents/skills/skill-creator/SKILL.md",
    });
    const bundled = buildSkill({ source: "public" });

    expect(getSkillKey(local)).not.toBe(getSkillKey(bundled));
  });

  it("distinguishes two local skills with the same name at different paths", () => {
    const personal = buildSkill({
      source: "/Users/test/.agents/skills/skill-creator/SKILL.md",
    });
    const project = buildSkill({
      source: "/workspace/demo/.agents/skills/skill-creator/SKILL.md",
    });

    expect(getSkillKey(personal)).not.toBe(getSkillKey(project));
  });

  it("is stable for the same entry and falls back to the name without a source", () => {
    const skill = buildSkill({ source: "public" });

    expect(getSkillKey(skill)).toBe(getSkillKey({ ...skill }));
    expect(getSkillKey(buildSkill())).toBe("skill-creator");
    expect(getSkillKey(buildSkill({ source: "  " }))).toBe("skill-creator");
  });
});
