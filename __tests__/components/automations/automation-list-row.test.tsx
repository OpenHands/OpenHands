import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AutomationListRow } from "#/components/features/automations/automation-list-row";
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
  name: "GitHub PR Reviewer",
  prompt: "Review pull requests.",
  enabled: true,
  trigger: {
    type: "event",
    on: "pull_request.opened",
    source: "github",
  },
  repository: "acme/repo",
  model: "Claude",
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

describe("AutomationListRow", () => {
  it("renders title, trigger meta, and action icons in a two-line list row", () => {
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
    expect(screen.getByText("pull_request.opened")).toBeInTheDocument();
    expect(screen.getByText("GitHub")).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-pills-automation-1"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-run-now-automation-1"),
    ).toHaveAttribute("aria-label", "AUTOMATIONS$RUN_NOW");
    expect(screen.getByTestId("automation-run-now-automation-1")).toHaveClass(
      "size-8",
    );
  });

  it("shows last-run status, relative time, and a sparkline when insights are present", () => {
    const latestRun = createRun({
      started_at: new Date(Date.now() - 10 * 60_000).toISOString(),
      completed_at: new Date(Date.now() - 8 * 60_000).toISOString(),
    });

    render(
      <AutomationListRow
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

    expect(
      screen.getByTestId("automation-last-run-automation-1"),
    ).toHaveTextContent("AUTOMATIONS$DETAIL$TIME_MINUTES_AGO");
    expect(screen.getByTestId("run-status-icon-completed")).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-activity-automation-1"),
    ).toBeInTheDocument();
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
