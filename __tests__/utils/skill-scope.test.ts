import { describe, expect, it } from "vitest";
import type { SkillInfo } from "#/types/settings";
import { getSkillScope, groupSkillsByScope } from "#/utils/skill-scope";

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return {
    name: "test-skill",
    type: "knowledge",
    source: null,
    ...overrides,
  };
}

describe("getSkillScope", () => {
  it("classifies public catalog skills", () => {
    expect(
      getSkillScope(
        buildSkill({
          source: "/Users/test/.openhands/cache/skills/public-skills/skills/deno/SKILL.md",
        }),
      ),
    ).toBe("public");
    expect(getSkillScope(buildSkill({ source: "public" }))).toBe("public");
  });

  it("classifies personal user skills from home directories", () => {
    expect(
      getSkillScope(
        buildSkill({
          source: "/Users/test/.agents/skills/my-skill/SKILL.md",
        }),
      ),
    ).toBe("personal");
    expect(getSkillScope(buildSkill({ source: "user" }))).toBe("personal");
  });

  it("classifies project skills from the workspace", () => {
    const projectDir = "/workspace/project/agent-canvas";
    expect(
      getSkillScope(
        buildSkill({
          source: `${projectDir}/.agents/skills/default-tools/SKILL.md`,
        }),
        projectDir,
      ),
    ).toBe("project");
    expect(getSkillScope(buildSkill({ source: "project" }))).toBe("project");
  });
});

describe("groupSkillsByScope", () => {
  it("groups and sorts skills by scope", () => {
    const grouped = groupSkillsByScope([
      buildSkill({ name: "beta", source: "public" }),
      buildSkill({ name: "alpha", source: "user" }),
      buildSkill({
        name: "gamma",
        source: "/workspace/project/.agents/skills/gamma/SKILL.md",
      }),
    ], "/workspace/project");

    expect(grouped.public.map((skill) => skill.name)).toEqual(["beta"]);
    expect(grouped.personal.map((skill) => skill.name)).toEqual(["alpha"]);
    expect(grouped.project.map((skill) => skill.name)).toEqual(["gamma"]);
  });
});
