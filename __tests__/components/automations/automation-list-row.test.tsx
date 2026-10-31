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
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { count?: number | string }) =>
      options?.count === undefined ? key : `${key}:${options.count}`,
    i18n: { language: "en" },
  }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: vi.fn(), currentPath: "/" }),
}));

vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => ({
    canView: true,
    canManage: true,
    isLoading: false,
  }),
  useIsAutomationOwner: () => true,
}));

// The pinned package predates the `impact` field, so an entry carrying one is
// appended to the real catalog.
vi.mock("@openhands/extensions/automations", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@openhands/extensions/automations")>();
  return {
    ...actual,
    AUTOMATION_CATALOG: [
      ...actual.AUTOMATION_CATALOG,
      {
        id: "widget-checker",
        impact: {
          basis: "completed-runs",
          one: "1 widget check completed",
          other: "{{count}} widget checks completed",
        },
      },
    ] as typeof actual.AUTOMATION_CATALOG,
  };
});

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

  it("keeps tags in their own box that gives way before the name", () => {
    render(
      <AutomationListRow
        automation={{
          ...automation,
          tags: ["nightly-checks", "qa-alpha", "pending-in-input"],
        }}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    // jsdom has no layout, so pin the classes that give the tags only the room
    // the name leaves and clip them, instead of squeezing the name or
    // overflowing the row.
    const tags = screen.getByTestId("automation-list-row-tags-automation-1");
    expect(tags).toHaveClass("min-w-0", "flex-1", "overflow-hidden");
    const pills = screen.getAllByTestId("automation-tag");
    expect(pills).toHaveLength(3);
    pills.forEach((pill) => {
      expect(tags).toContainElement(pill);
    });
    expect(screen.getByTitle("nightly-checks")).toHaveTextContent(
      "nightly-checks",
    );
    expect(tags).not.toContainElement(screen.getByText("GitHub PR Reviewer"));
  });

  it("renders no tag box for an automation without tags", () => {
    render(
      <AutomationListRow
        automation={{ ...automation, tags: [] }}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(
      screen.queryByTestId("automation-list-row-tags-automation-1"),
    ).not.toBeInTheDocument();
  });

  it("marks draft automations separately from inactive rows", () => {
    render(
      <AutomationListRow
        automation={{ ...automation, enabled: false, state: "DRAFT" }}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(screen.getByTestId("active-status-badge-draft")).toHaveTextContent(
      I18nKey.AUTOMATIONS$DETAIL$DRAFT,
    );
    expect(
      screen.queryByTestId("active-status-badge-inactive"),
    ).not.toBeInTheDocument();
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
              completedTotal: 4,
              pendingTotal: 0,
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
    expect(
      screen.queryByTestId("automation-pending-automation-1"),
    ).not.toBeInTheDocument();
  });

  it("shows the queued run count in the meta line", () => {
    // Arrange
    const latestRun = createRun({
      status: AutomationRunStatus.PENDING,
      completed_at: null,
    });

    // Act
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
              total: 7,
              completedTotal: 4,
              pendingTotal: 3,
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

    // Assert
    expect(
      screen.getByTestId("automation-pending-automation-1"),
    ).toHaveTextContent("AUTOMATIONS$PENDING_RUN_COUNT:3");
  });

  it("shows the value statement in the meta line", () => {
    // Arrange — provenance joining back to a catalog entry with an impact
    // declaration, and a summary carrying the lifetime completed count.
    const latestRun = createRun();

    // Act
    render(
      <AutomationListRow
        automation={{
          ...automation,
          preset_metadata: {
            template: { id: "widget-checker", version: "1.0.0", config: {} },
          },
        }}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
        insights={{
          spec: insightsSpec satisfies InterfaceListInsights,
          state: {
            summary: {
              total: 5,
              completedTotal: 4,
              pendingTotal: 0,
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

    // Assert
    expect(
      screen.getByTestId("automation-impact-automation-1"),
    ).toHaveTextContent("4 widget checks completed");
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
