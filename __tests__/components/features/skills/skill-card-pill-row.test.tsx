import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  SKILL_CARD_PILL_CLASS,
  SkillCardPillRow,
} from "#/components/features/skills/skill-card-pill-row";

describe("SkillCardPillRow", () => {
  it("keeps pills on a single nowrap row with overflow handling", () => {
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}

        disconnect() {}
      },
    );

    render(
      <SkillCardPillRow
        testId="skill-triggers-test"
        pills={[
          {
            id: "type-knowledge",
            node: <span className={SKILL_CARD_PILL_CLASS}>Trigger-based</span>,
          },
          {
            id: "trigger-ssh",
            node: <span className={SKILL_CARD_PILL_CLASS}>ssh</span>,
          },
        ]}
      />,
    );

    const row = screen.getByTestId("skill-triggers-test");
    expect(row).toHaveClass("flex-nowrap");
    expect(row).toHaveClass("overflow-hidden");
    expect(row).not.toHaveClass("flex-wrap");
  });
});
