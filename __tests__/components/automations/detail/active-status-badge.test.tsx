import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ActiveStatusBadge } from "#/components/features/automations/detail/active-status-badge";
import { I18nKey } from "#/i18n/declaration";
import {
  LIGHT_THEMES,
  PANEL_SURFACES,
  colorUtility,
  contrast,
  over,
  surfaceColor,
  utilityColor,
} from "../../../themes/theme-color-resolver";

describe("ActiveStatusBadge", () => {
  it.each(LIGHT_THEMES)(
    "keeps the rendered draft badge readable on %s panels",
    (theme) => {
      render(
        <ActiveStatusBadge automation={{ enabled: false, state: "DRAFT" }} />,
      );
      const classes = screen.getByTestId("active-status-badge-draft").className;
      const fill = utilityColor(theme, colorUtility(classes, "bg"));
      const ink = utilityColor(theme, colorUtility(classes, "text"));
      for (const surface of PANEL_SURFACES) {
        const background = over(fill, surfaceColor(theme, surface));
        expect
          .soft(contrast(over(ink, background), background), surface)
          .toBeGreaterThanOrEqual(4.5);
      }
    },
  );
  it.each([
    [
      { enabled: true },
      I18nKey.AUTOMATIONS$DETAIL$ACTIVE,
      "active-status-badge-active",
    ],
    [
      { enabled: false },
      I18nKey.AUTOMATIONS$DETAIL$INACTIVE,
      "active-status-badge-inactive",
    ],
    [
      { enabled: false, state: "DRAFT" },
      I18nKey.AUTOMATIONS$DETAIL$DRAFT,
      "active-status-badge-draft",
    ],
  ])(
    "renders the matching label and testid for automation %o",
    (automation, labelKey, testId) => {
      render(<ActiveStatusBadge automation={automation} />);

      expect(screen.getByText(labelKey)).toBeInTheDocument();
      expect(screen.getByTestId(testId)).toBeInTheDocument();
    },
  );
});
