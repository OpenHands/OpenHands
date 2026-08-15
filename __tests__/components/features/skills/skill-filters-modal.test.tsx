import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { SkillFiltersModal } from "#/components/features/skills/skill-filters-modal";
import type { SkillFacetGroup } from "#/components/features/skills/skill-filter";

function buildGroups(): SkillFacetGroup[] {
  return [
    {
      id: "type",
      labelKey: I18nKey.SETTINGS$SKILLS_FACET_TYPE,
      rows: [
        {
          value: "repo",
          labelKey: I18nKey.SETTINGS$SKILLS_TYPE_REPO,
          count: 1,
          checked: false,
          disabled: false,
        },
        {
          value: "knowledge",
          labelKey: I18nKey.SETTINGS$SKILLS_TYPE_KNOWLEDGE,
          count: 2,
          checked: false,
          disabled: false,
        },
      ],
    },
  ];
}

describe("SkillFiltersModal", () => {
  it("toggles a facet through the same rail the desktop uses", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();

    render(
      <SkillFiltersModal
        groups={buildGroups()}
        activeCount={0}
        onToggle={onToggle}
        onClearAll={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("skill-facet-type-repo"));

    expect(onToggle).toHaveBeenCalledWith("type", "repo");
  });

  it("hides the clear action when nothing is filtered", () => {
    render(
      <SkillFiltersModal
        groups={buildGroups()}
        activeCount={0}
        onToggle={vi.fn()}
        onClearAll={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(
      screen.queryByTestId("skill-filters-modal-clear"),
    ).not.toBeInTheDocument();
  });

  it("clears filters and closes on demand", async () => {
    const user = userEvent.setup();
    const onClearAll = vi.fn();
    const onClose = vi.fn();

    render(
      <SkillFiltersModal
        groups={buildGroups()}
        activeCount={2}
        onToggle={vi.fn()}
        onClearAll={onClearAll}
        onClose={onClose}
      />,
    );

    await user.click(screen.getByTestId("skill-filters-modal-clear"));
    expect(onClearAll).toHaveBeenCalled();

    await user.click(screen.getByTestId("skill-filters-modal-done"));
    expect(onClose).toHaveBeenCalled();
  });
});
