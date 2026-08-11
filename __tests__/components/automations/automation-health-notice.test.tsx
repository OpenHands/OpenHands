import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutomationHealthNotice } from "#/components/features/automations/automation-health-notice";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

describe("AutomationHealthNotice", () => {
  it("shows an auto-disable reason and offers to turn the automation back on", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();

    render(
      <AutomationHealthNotice
        automation={{ id: "automation-1", enabled: false }}
        canManage
        details={{
          issue: "disabled",
          failureKind: "config",
          reason: "Model profile is no longer available",
        }}
        onToggle={onToggle}
        onView={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("automation-health-notice-automation-1"),
    ).toHaveTextContent("Model profile is no longer available");
    expect(
      screen.getByTestId("automation-health-notice-automation-1"),
    ).toHaveTextContent("(Config)");

    await user.click(
      screen.getByRole("button", { name: "AUTOMATIONS$TURN_ON" }),
    );

    expect(onToggle).toHaveBeenCalledWith("automation-1", false);
  });

  it("shows a transient error and links to the automation details", async () => {
    const user = userEvent.setup();
    const onView = vi.fn();

    render(
      <AutomationHealthNotice
        automation={{ id: "automation-2", enabled: true }}
        canManage={false}
        details={{
          issue: "transient",
          failureKind: "rate_limit",
          reason: "The provider is rate limiting requests",
        }}
        onToggle={vi.fn()}
        onView={onView}
      />,
    );

    expect(
      screen.getByText("The provider is rate limiting requests"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "COMMON$VIEW" }));
    expect(onView).toHaveBeenCalledOnce();
  });

  it("keeps disabled and transient notices visually distinct", () => {
    const { rerender } = render(
      <AutomationHealthNotice
        automation={{ id: "automation-4", enabled: false }}
        canManage
        details={{
          issue: "disabled",
          failureKind: "config",
          reason: "The selected model profile is no longer available.",
        }}
        onToggle={vi.fn()}
        onView={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("automation-health-notice-automation-4"),
    ).toHaveClass("bg-surface-raised");

    rerender(
      <AutomationHealthNotice
        automation={{ id: "automation-4", enabled: true }}
        canManage={false}
        details={{
          issue: "transient",
          failureKind: "rate_limit",
          reason: "The provider is rate limiting requests",
        }}
        onToggle={vi.fn()}
        onView={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("automation-health-notice-automation-4"),
    ).toHaveClass("bg-[var(--oh-warning)]/10");
  });

  it("renders nothing when there is no actionable health detail", () => {
    const { container } = render(
      <AutomationHealthNotice
        automation={{ id: "automation-3", enabled: true }}
        canManage
        details={{ issue: null, failureKind: null, reason: null }}
        onToggle={vi.fn()}
        onView={vi.fn()}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
