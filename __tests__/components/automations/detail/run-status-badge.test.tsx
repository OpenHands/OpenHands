import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { AutomationRunStatus } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";

describe("RunStatusBadge", () => {
  it.each([
    [AutomationRunStatus.COMPLETED, I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL],
    [AutomationRunStatus.FAILED, I18nKey.AUTOMATIONS$DETAIL$FAILED],
    [AutomationRunStatus.CANCELLED, I18nKey.AUTOMATIONS$DETAIL$CANCELLED],
    [AutomationRunStatus.SKIPPED, I18nKey.AUTOMATIONS$DETAIL$SKIPPED],
    [AutomationRunStatus.PENDING, I18nKey.AUTOMATIONS$DETAIL$PENDING],
    [AutomationRunStatus.RUNNING, I18nKey.AUTOMATIONS$DETAIL$RUNNING],
  ])("renders the %s label for the matching status", (status, labelKey) => {
    render(<RunStatusBadge status={status} />);

    expect(screen.getByText(labelKey)).toBeInTheDocument();
  });

  it.each([
    [AutomationRunStatus.COMPLETED, "run-status-icon-completed"],
    [AutomationRunStatus.FAILED, "run-status-icon-failed"],
    [AutomationRunStatus.CANCELLED, "run-status-icon-cancelled"],
    [AutomationRunStatus.SKIPPED, "run-status-icon-skipped"],
    [AutomationRunStatus.PENDING, "run-status-icon-pending"],
    [AutomationRunStatus.RUNNING, "run-status-icon-pending"],
  ])(
    "renders the %s icon variant for the matching status",
    (status, testId) => {
      render(<RunStatusBadge status={status} />);

      expect(screen.getByTestId(testId)).toBeInTheDocument();
    },
  );

  it("falls back to a pending badge for an unknown backend status", () => {
    render(<RunStatusBadge status={"UNKNOWN" as AutomationRunStatus} />);

    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$PENDING),
    ).toBeInTheDocument();
    expect(screen.getByTestId("run-status-icon-pending")).toBeInTheDocument();
  });
});
