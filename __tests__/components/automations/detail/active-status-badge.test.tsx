import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ActiveStatusBadge } from "#/components/features/automations/detail/active-status-badge";
import { I18nKey } from "#/i18n/declaration";

describe("ActiveStatusBadge", () => {
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
