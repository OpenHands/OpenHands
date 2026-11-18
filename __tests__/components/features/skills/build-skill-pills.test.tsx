import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { type TFunction } from "i18next";
import { buildSkillPills } from "#/components/features/skills/build-skill-pills";
import { type SkillCardPill } from "#/components/features/skills/skill-card-pill-row";
import { type SkillInfo } from "#/types/settings";

import { getSkillCardDescription } from "#/components/features/skills/get-skill-card-description";
import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import {
  ADD_SKILL_EXAMPLE_COMMAND,
  ADD_SKILL_SKILL_NAME,
} from "#/constants/skills-docs";
import { getSkillChatLaunchMessage } from "#/components/features/skills/get-skill-chat-launch-message";
import { isCopyableSkillSource } from "#/components/features/skills/is-copyable-skill-source";

describe("build-skill-pills", () => {
  /** Returns its key unchanged, for assertions that do not care about copy. */
  const translate = ((key: string) => key) as TFunction;

  function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
    return {
      name: "deno",
      type: "knowledge",
      source: "public",
      category: "environment",
      triggers: [],
      ...overrides,
    };
  }

  /** Mounts the built pills so their rendered output can be asserted. */
  function renderPills(pills: SkillCardPill[]) {
    render(
      <div>
        {pills.map((pill) => (
          <span key={pill.id}>{pill.node}</span>
        ))}
      </div>,
    );
  }

  describe("buildSkillPills category pill", () => {
    it("renders the catalog category", () => {
      renderPills(
        buildSkillPills(buildSkill({ category: "environment" }), translate),
      );

      expect(screen.getByTestId("skill-category-deno")).toHaveTextContent(
        "SETTINGS$SKILLS_CATEGORY_ENVIRONMENT",
      );
    });

    it("omits the pill when the skill is uncategorized", () => {
      renderPills(buildSkillPills(buildSkill({ category: null }), translate));

      expect(
        screen.queryByTestId("skill-category-deno"),
      ).not.toBeInTheDocument();
    });

    it("places the category pill directly after the type badge", () => {
      // Order is asserted on pill ids, so no rendering is needed here.
      const pills = buildSkillPills(buildSkill(), translate);

      expect(pills[0]!.id).toBe("type-knowledge");
      expect(pills[1]!.id).toBe("category-environment");
    });
  });
});

describe("get-skill-card-description", () => {
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
});

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

describe("isCopyableSkillSource", () => {
  it("returns false for scope labels and empty values", () => {
    expect(isCopyableSkillSource(null)).toBe(false);
    expect(isCopyableSkillSource("")).toBe(false);
    expect(isCopyableSkillSource("global")).toBe(false);
    expect(isCopyableSkillSource("agent")).toBe(false);
  });

  it("returns true for filesystem paths and URLs", () => {
    expect(isCopyableSkillSource("/skills/deno/SKILL.md")).toBe(true);
    expect(isCopyableSkillSource(".agents/skills/foo/SKILL.md")).toBe(true);
    expect(isCopyableSkillSource("https://github.com/example/skills")).toBe(
      true,
    );
  });
});
