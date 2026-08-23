import { describe, expect, it } from "vitest";
import { SKILL_CATEGORY_IDS } from "@openhands/extensions/skills";
import type { SkillInfo } from "#/types/settings";
import {
  getSkillCategory,
  SKILL_CATEGORY_ICONS,
  SKILL_CATEGORY_LABEL_KEYS,
  SKILL_CATEGORY_ORDER,
} from "#/utils/skill-category";

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return { name: "test-skill", type: "knowledge", source: null, ...overrides };
}

describe("skill-category", () => {
  it("returns the catalog category when it is known", () => {
    expect(getSkillCategory(buildSkill({ category: "environment" }))).toBe(
      "environment",
    );
  });

  it.each([
    ["missing", undefined],
    ["null", null],
    ["unknown to this build", "code-hostig" as never],
  ])("falls back to other when the category is %s", (_label, category) => {
    expect(getSkillCategory(buildSkill({ category }))).toBe("other");
  });

  it("covers every catalog category in the rail order, labels, and icons", () => {
    const ids = [...SKILL_CATEGORY_IDS].sort();
    expect([...SKILL_CATEGORY_ORDER].sort()).toEqual(ids);
    expect(Object.keys(SKILL_CATEGORY_LABEL_KEYS).sort()).toEqual(ids);
    expect(Object.keys(SKILL_CATEGORY_ICONS).sort()).toEqual(ids);
  });
});
