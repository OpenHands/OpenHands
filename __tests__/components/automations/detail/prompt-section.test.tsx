import { describe, expect, it } from "vitest";
import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { I18nKey } from "#/i18n/declaration";
import { PromptSection } from "#/components/features/automations/detail/prompt-section";
import { RunStatusBadge } from "#/components/features/automations/detail/run-status-badge";
import { AutomationRunStatus } from "#/types/automation";

describe("PromptSection", () => {
  it("does not show a toggle for short prompts", () => {
    render(<PromptSection prompt="Short prompt." />);

    expect(screen.getByTestId("automation-prompt-content")).toHaveTextContent(
      "Short prompt.",
    );
    expect(
      screen.queryByTestId("automation-prompt-toggle"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-prompt-fade"),
    ).not.toBeInTheDocument();
  });

  it("collapses long prompts with a fade and expands on toggle", () => {
    const longPrompt = Array.from(
      { length: 40 },
      (_, index) => `Line ${index + 1}.`,
    ).join("\n");

    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get() {
        return this.textContent?.includes("Line 40.") ? 480 : 120;
      },
    });

    render(<PromptSection prompt={longPrompt} />);

    expect(screen.getByTestId("automation-prompt-fade")).toBeInTheDocument();
    expect(screen.getByTestId("automation-prompt-content")).toHaveClass(
      "max-h-60",
    );
    expect(screen.getByTestId("automation-prompt-toggle")).toHaveTextContent(
      I18nKey.COMMON$VIEW_MORE,
    );

    fireEvent.click(screen.getByTestId("automation-prompt-toggle"));

    expect(
      screen.queryByTestId("automation-prompt-fade"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("automation-prompt-content")).not.toHaveClass(
      "max-h-60",
    );
    expect(screen.getByTestId("automation-prompt-toggle")).toHaveTextContent(
      I18nKey.COMMON$VIEW_LESS,
    );
  });
});

describe("RunStatusBadge", () => {
  it.each([
    [AutomationRunStatus.COMPLETED, I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL],
    [AutomationRunStatus.FAILED, I18nKey.AUTOMATIONS$DETAIL$FAILED],
    [AutomationRunStatus.PENDING, I18nKey.AUTOMATIONS$DETAIL$PENDING],
    [AutomationRunStatus.RUNNING, I18nKey.AUTOMATIONS$DETAIL$RUNNING],
    [AutomationRunStatus.CANCELLED, I18nKey.AUTOMATIONS$DETAIL$CANCELLED],
    [AutomationRunStatus.SKIPPED, I18nKey.AUTOMATIONS$DETAIL$SKIPPED],
    ["success", I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL],
    ["blocked", I18nKey.AUTOMATIONS$DETAIL$BLOCKED],
    ["failed", I18nKey.AUTOMATIONS$DETAIL$FAILED],
    ["partial_success", I18nKey.AUTOMATIONS$DETAIL$PARTIAL],
    ["unknown", I18nKey.AUTOMATIONS$DETAIL$NEEDS_REVIEW],
  ] as const)(
    "renders the %s label for the matching status",
    (status, labelKey) => {
      render(<RunStatusBadge status={status} />);

      expect(screen.getByText(labelKey)).toBeInTheDocument();
    },
  );

  it.each([
    [AutomationRunStatus.COMPLETED, "run-status-icon-completed"],
    [AutomationRunStatus.FAILED, "run-status-icon-failed"],
    [AutomationRunStatus.PENDING, "run-status-icon-pending"],
    [AutomationRunStatus.RUNNING, "run-status-icon-running"],
    [AutomationRunStatus.CANCELLED, "run-status-icon-pending"],
    [AutomationRunStatus.SKIPPED, "run-status-icon-pending"],
    ["success", "run-status-icon-completed"],
    ["blocked", "run-status-icon-warning"],
    ["failed", "run-status-icon-failed"],
    ["partial_success", "run-status-icon-warning"],
    ["unknown", "run-status-icon-needs-review"],
  ] as const)(
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
    render(<RunStatusBadge status={AutomationRunStatus.COMPLETED} iconOnly />);

    expect(
      screen.queryByText(I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL),
    ).toBeNull();
    expect(screen.getByTestId("run-status-badge-icon")).toHaveAttribute(
      "aria-label",
      I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL,
    );
    expect(screen.getByTestId("run-status-icon-completed")).toBeInTheDocument();
  });

  it("renders compact pills without an outline and with tighter left padding", () => {
    render(<RunStatusBadge status={AutomationRunStatus.FAILED} compact />);

    const badge = screen.getByText(I18nKey.AUTOMATIONS$DETAIL$FAILED);
    expect(badge.className).not.toContain("border");
    expect(badge).toHaveClass("pl-1");
    expect(badge).toHaveClass("pr-1.5");
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
