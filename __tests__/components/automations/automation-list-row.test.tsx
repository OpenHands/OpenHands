import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutomationListRow } from "#/components/features/automations/automation-list-row";
import type { Automation } from "#/types/automation";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: vi.fn() }),
}));

vi.mock("#/hooks/use-has-permission", () => ({
  useHasPermission: () => true,
}));

const automation: Automation = {
  id: "automation-1",
  name: "GitHub PR Reviewer",
  prompt: "Review pull requests.",
  enabled: true,
  trigger: { type: "event" },
  repository: "acme/repo",
  model: "Claude",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

describe("AutomationListRow", () => {
  it("renders title, pills, and action icons in a table row layout", () => {
    render(
      <AutomationListRow
        automation={automation}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("automation-list-row-automation-1"),
    ).toBeInTheDocument();
    expect(screen.getByText("GitHub PR Reviewer")).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-pills-automation-1"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-run-now-automation-1"),
    ).toHaveAttribute("aria-label", "AUTOMATIONS$RUN_NOW");
    expect(screen.getByTestId("automation-run-now-automation-1")).toHaveClass(
      "size-8",
    );
  });

  it("opens the actions menu without triggering row navigation handlers", async () => {
    const user = userEvent.setup();

    render(
      <AutomationListRow
        automation={automation}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: "AUTOMATIONS$ACTIONS_MENU" }),
    );

    expect(screen.getByText("COMMON$VIEW")).toBeInTheDocument();
  });
});
