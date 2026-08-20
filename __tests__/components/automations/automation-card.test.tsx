import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutomationCard } from "#/components/features/automations/automation-card";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import type { InterfaceListInsights } from "#/manifests/types";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { language: "en" },
  }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: vi.fn(), currentPath: "/" }),
}));

vi.mock("#/hooks/use-has-permission", () => ({
  useHasPermission: () => true,
}));

const automation: Automation = {
  id: "automation-1",
  name: "Async Standup Digest",
  prompt: "Generate an async standup digest from Slack activity.",
  enabled: true,
  trigger: { type: "cron", schedule_human: "Mondays at 09:00" },
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const insightsSpec = {
  health: {
    healthy: "Healthy",
    failing: "Failing",
    running: "Running",
    disabled: "Disabled",
    neverRun: "Never run",
    checking: "Checking",
  },
  lastRun: { label: "Last run", never: "Never", justNow: "Just now" },
  stats: { runs: "Runs", recentSuccess: "Success", averageDuration: "Avg" },
};

function createRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-01-02T00:00:00Z",
    completed_at: "2026-01-02T00:02:00Z",
    ...overrides,
  };
}

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
    expect(card.className).toContain("bg-base-secondary");
    expect(card.className).not.toContain("border-[var(--oh-border)]");
  });

  it("renders title, description, and overflow pills", () => {
    render(
      <AutomationCard
        automation={automation}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(screen.getByText("Async Standup Digest")).toBeInTheDocument();
    expect(
      screen.getByText("Generate an async standup digest from Slack activity."),
    ).toBeInTheDocument();
    expect(screen.getByText("Mondays at 09:00")).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-pills-automation-1"),
    ).toBeInTheDocument();
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
    ).toHaveAttribute("aria-label", "AUTOMATIONS$RUN_NOW");
    expect(screen.getByTestId("automation-run-now-automation-1")).toHaveClass(
      "size-8",
    );
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "AUTOMATIONS$ACTIONS_MENU" }),
    );

    expect(screen.getByText("COMMON$VIEW")).toBeInTheDocument();
    expect(screen.getByText("AUTOMATIONS$RUN_NOW")).toBeInTheDocument();
  });

  it("shows a status strip and sparkline when insights are present", () => {
    const latestRun = createRun({
      started_at: new Date(Date.now() - 10 * 60_000).toISOString(),
      completed_at: new Date(Date.now() - 8 * 60_000).toISOString(),
    });

    render(
      <AutomationCard
        automation={automation}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
        insights={{
          spec: insightsSpec satisfies InterfaceListInsights,
          state: {
            summary: {
              total: 4,
              latestRun,
              recentRuns: [latestRun],
              recentSuccessRate: 1,
              averageDurationMs: 120_000,
            },
            isLoading: false,
            isError: false,
          },
        }}
      />,
    );

    expect(screen.queryByTestId("automation-health-badge")).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-last-run-automation-1"),
    ).toHaveTextContent("AUTOMATIONS$DETAIL$TIME_MINUTES_AGO");
    expect(screen.getByTestId("run-status-icon-completed")).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-activity-automation-1"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("automation-run-stats")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
  });
});
