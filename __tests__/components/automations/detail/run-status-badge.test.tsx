import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { AutomationRunStatus } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";

describe("RunStatusBadge", () => {
  it.each([
    [AutomationRunStatus.COMPLETED, I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL],
    [AutomationRunStatus.FAILED, I18nKey.AUTOMATIONS$DETAIL$FAILED],
    [AutomationRunStatus.PENDING, I18nKey.AUTOMATIONS$DETAIL$PENDING],
    [AutomationRunStatus.RUNNING, I18nKey.AUTOMATIONS$DETAIL$RUNNING],
    [AutomationRunStatus.CANCELLED, I18nKey.AUTOMATIONS$DETAIL$CANCELLED],
    [AutomationRunStatus.SKIPPED, I18nKey.AUTOMATIONS$DETAIL$SKIPPED],
  ])("renders the %s label for the matching status", (status, labelKey) => {
    render(<RunStatusBadge status={status} />);

    expect(screen.getByText(labelKey)).toBeInTheDocument();
  });

  it.each([
    [AutomationRunStatus.COMPLETED, "run-status-icon-completed"],
    [AutomationRunStatus.FAILED, "run-status-icon-failed"],
    [AutomationRunStatus.PENDING, "run-status-icon-pending"],
    [AutomationRunStatus.RUNNING, "run-status-icon-running"],
    [AutomationRunStatus.CANCELLED, "run-status-icon-pending"],
    [AutomationRunStatus.SKIPPED, "run-status-icon-pending"],
  ])(
    "renders the %s icon variant for the matching status",
    (status, testId) => {
      render(<RunStatusBadge status={status} />);

      expect(screen.getByTestId(testId)).toBeInTheDocument();
    },
  );

  it("renders a neutral badge for a status the backend added later", () => {
    const unknownStatus = "ARCHIVED" as AutomationRunStatus;

    render(<RunStatusBadge status={unknownStatus} />);

    expect(screen.getByTestId("run-status-icon-pending")).toBeInTheDocument();
  });

  it("renders an icon-only mark with the status label as aria-label", () => {
    render(
      <RunStatusBadge status={AutomationRunStatus.COMPLETED} iconOnly />,
    );

    expect(screen.queryByText(I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL)).toBeNull();
    expect(screen.getByTestId("run-status-badge-icon")).toHaveAttribute(
      "aria-label",
      I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL,
    );
    expect(screen.getByTestId("run-status-icon-completed")).toBeInTheDocument();
  });

  it("renders the status word next to the icon when showLabel is set", () => {
    render(
      <RunStatusBadge
        status={AutomationRunStatus.PENDING}
        iconOnly
        showLabel
      />,
    );

    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$PENDING),
    ).toBeInTheDocument();
    expect(screen.getByTestId("run-status-icon-pending")).toBeInTheDocument();
  });
});
