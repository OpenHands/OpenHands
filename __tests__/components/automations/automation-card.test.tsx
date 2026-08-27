import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { AutomationCard } from "#/components/features/automations/automation-card";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import { useAutomationRunSummaries } from "#/hooks/query/use-automation-run-summaries";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";
import { server } from "#/mocks/node";
import { I18nKey } from "#/i18n/declaration";
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
              completedTotal: 4,
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
      screen.queryByTestId("automation-health-badge"),
    ).not.toBeInTheDocument();
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

  it("shows the value statement for its completed runs", () => {
    // Arrange — provenance joining back to a catalog entry with an impact
    // declaration, and a summary carrying the lifetime completed count.
    const latestRun = createRun();

    // Act
    render(
      <AutomationCard
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
});

describe("AutomationCard — run phase", () => {
  const localBackend: Backend = {
    id: "local-1",
    name: "Local 1",
    host: "http://localhost:8000",
    apiKey: "k",
    kind: "local",
  };

  const insightAutomation: Automation = {
    id: "auto-with-active-run",
    name: "Digest",
    prompt: null,
    enabled: true,
    trigger: { type: "cron", schedule_human: "cron" },
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };

  const insightsSpec: InterfaceListInsights = {
    health: {
      healthy: "Healthy",
      failing: "Failing",
      running: "Running",
      disabled: "Disabled",
      neverRun: "Never run",
      checking: "Checking",
    },
    lastRun: { label: "Last run", never: "Never", justNow: "Just now" },
    stats: {
      runs: "Runs",
      recentSuccess: "Success",
      averageDuration: "Duration",
    },
  };

  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  function Harness() {
    const byId = useAutomationRunSummaries([insightAutomation]);
    return (
      <AutomationCard
        automation={insightAutomation}
        onToggle={vi.fn()}
        onRunNow={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
        insights={{ spec: insightsSpec, state: byId.get(insightAutomation.id) }}
      />
    );
  }

  function renderHarness() {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    return render(
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <Harness />
        </ActiveBackendProvider>
      </QueryClientProvider>,
    );
  }

  it("shows the active run's phase using only the run-summaries fetch insights already makes — no extra request", async () => {
    // Arrange: count every hit to the runs endpoint the card's insights
    // already fetch (via useAutomationRunSummaries) — if displaying the
    // phase required a second request, this would be > 1.
    let callCount = 0;
    server.use(
      http.get("*/api/automation/v1/:id/runs", () => {
        callCount += 1;
        return HttpResponse.json({
          runs: [
            {
              id: "run-active",
              status: AutomationRunStatus.RUNNING,
              conversation_id: null,
              bash_command_id: null,
              error_detail: null,
              phase_code: "running_agent",
              phase_label: null,
              phase_updated_at: null,
              started_at: "2026-01-01T09:00:00Z",
              completed_at: null,
            },
          ],
          total: 1,
        });
      }),
    );

    // Act
    renderHarness();

    // Assert: the phase renders ...
    await screen.findByText(I18nKey.AUTOMATIONS$DETAIL$PHASE_RUNNING_AGENT);
    // ... and exactly one request was made — the pre-existing insights
    // fetch, not a new one just for the phase.
    expect(callCount).toBe(1);
  });
});
