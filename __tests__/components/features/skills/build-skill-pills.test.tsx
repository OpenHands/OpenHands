import { render, screen } from "@testing-library/react";
import { describe, expect, it, beforeAll } from "vitest";
import type { TFunction } from "i18next";
import { buildSkillPills } from "#/components/features/skills/build-skill-pills";
import type { SkillInfo } from "#/types/settings";
import { translationResources } from "#/i18n/resources";
import i18n from "#/i18n";

/** Returns its key unchanged, for assertions that do not care about copy. */
const identityTranslate = ((key: string) => key) as unknown as TFunction;

beforeAll(async () => {
  // Configure i18n with resources for testing
  i18n.addResourceBundle("en", "openhands", translationResources.en || {});
});

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

/** Renders the built pills so translated output can be asserted. */
function PillHarness({ skill }: { skill: SkillInfo }) {
  const t = (key: string, options?: Record<string, unknown>) =>
    i18n.t(key, options);
  return (
    <div>
      {buildSkillPills(skill, t as TFunction).map((pill) => (
        <span key={pill.id}>{pill.node}</span>
      ))}
    </div>
  );
}

describe("buildSkillPills category pill", () => {
  it("renders the catalog category", () => {
    render(<PillHarness skill={buildSkill({ category: "environment" })} />);

    expect(screen.getByTestId("skill-category-deno")).toHaveTextContent(
      "Environment & tooling",
    );
  });

  it("renders Uncategorized when the skill has no category", () => {
    render(<PillHarness skill={buildSkill({ category: null })} />);

    expect(screen.getByTestId("skill-category-deno")).toHaveTextContent(
      "Uncategorized",
    );
  });

  it("places the category pill directly after the type badge", () => {
    // Order is asserted on pill ids, so a pass-through translate is enough —
    // no rendering and no i18n setup needed.
    const pills = buildSkillPills(buildSkill(), identityTranslate);

    expect(pills[0]!.id).toBe("type-knowledge");
    expect(pills[1]!.id).toBe("category-environment");
  });
});
