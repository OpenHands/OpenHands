import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutomationCard } from "#/components/features/automations/automation-card";
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
  name: "Async Standup Digest",
  prompt: "Generate an async standup digest from Slack activity.",
  enabled: true,
  trigger: { type: "cron", schedule_human: "cron" },
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

describe("AutomationCard", () => {
  it("uses the shared extension module interactive class without a resting border", () => {
    render(
      <AutomationCard
        automation={automation}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    const card = screen.getByTestId("automation-card-automation-1");
    expect(card.className).toContain("extension-module-card-interactive");
    expect(card.className).not.toContain("border-[var(--oh-border)]");
    expect(card.className).not.toContain("hover:bg-surface-raised");
    expect(card.className).not.toContain("hover:ring");
  });

  it("renders a play run button and menu actions instead of a toggle switch", async () => {
    const user = userEvent.setup();

    render(
      <AutomationCard
        automation={automation}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("automation-run-now-automation-1"),
    ).toHaveTextContent("AUTOMATIONS$RUN_NOW");
    expect(screen.getByTestId("automation-run-now-automation-1")).toHaveClass(
      "h-8",
    );
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "AUTOMATIONS$ACTIONS_MENU" }),
    );

    expect(screen.getByText("COMMON$VIEW")).toBeInTheDocument();
    expect(screen.getAllByText("AUTOMATIONS$RUN_NOW")).toHaveLength(2);
  });
});
