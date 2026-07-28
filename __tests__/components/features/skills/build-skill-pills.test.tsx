import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TFunction } from "i18next";
import { buildSkillPills } from "#/components/features/skills/build-skill-pills";
import type { SkillCardPill } from "#/components/features/skills/skill-card-pill-row";
import type { SkillInfo } from "#/types/settings";

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

    expect(screen.queryByTestId("skill-category-deno")).not.toBeInTheDocument();
  });

  it("places the category pill directly after the type badge", () => {
    // Order is asserted on pill ids, so no rendering is needed here.
    const pills = buildSkillPills(buildSkill(), translate);

    expect(pills[0]!.id).toBe("type-knowledge");
    expect(pills[1]!.id).toBe("category-environment");
  });
});
