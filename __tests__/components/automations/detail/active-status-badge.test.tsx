import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ActiveStatusBadge } from "#/components/features/automations/detail/active-status-badge";
import { I18nKey } from "#/i18n/declaration";

describe("ActiveStatusBadge", () => {
  it.each([
    [true, I18nKey.AUTOMATIONS$DETAIL$ACTIVE, "active-status-badge-active"],
    [
      false,
      I18nKey.AUTOMATIONS$DETAIL$INACTIVE,
      "active-status-badge-inactive",
    ],
  ])(
    "renders the matching label and testid when active=%s",
    (active, labelKey, testId) => {
      render(<ActiveStatusBadge active={active} />);

      expect(screen.getByText(labelKey)).toBeInTheDocument();
      expect(screen.getByTestId(testId)).toBeInTheDocument();
    },
  );
});
