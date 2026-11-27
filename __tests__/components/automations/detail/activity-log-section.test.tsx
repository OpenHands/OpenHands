import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ActivityLogSection } from "#/components/features/automations/detail/activity-log-section";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import { downloadActivityLogExport } from "#/utils/automation-activity-log-export";

const { trackAutomationActivityLogExported } = vi.hoisted(() => ({
  trackAutomationActivityLogExported: vi.fn(),
}));

vi.mock("#/utils/automation-activity-log-export", () => ({
  downloadActivityLogExport: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackAutomationActivityLogExported,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "default-local", kind: "local" },
    orgId: null,
  }),
}));

vi.mock("#/hooks/query/use-automation-detail", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/hooks/query/use-automation-detail")
  >()),
  useAutomationRuns: vi.fn(),
}));

// The run stats read the same sample through the service.
vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomationRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
  },
}));

import { useAutomationRuns } from "#/hooks/query/use-automation-detail";

const automation: Automation = {
  id: "a1",
  name: "Test Activity Log",
  trigger: { type: "cron", schedule: "0 9 * * 1-5", timezone: "UTC" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  prompt: "hello",
};

const run: AutomationRun = {
  id: "r1",
  status: AutomationRunStatus.COMPLETED,
  conversation_id: "c1",
  bash_command_id: "b1",
  error_detail: null,
  started_at: "2026-01-01T09:00:00Z",
  completed_at: "2026-01-01T09:01:00Z",
};

function renderSection(highlightedRunId: string | null = null) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActivityLogSection
        automation={automation}
        highlightedRunId={highlightedRunId}
      />
    </QueryClientProvider>,
  );
}

describe("ActivityLogSection export", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: [run], total: 1 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);
  });

  it("exports JSON via the activity-log export helper and tracks it", async () => {
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId("activity-log-export-json"));

    await waitFor(() => {
      expect(downloadActivityLogExport).toHaveBeenCalledWith({
        automation,
        format: "json",
        conversationBaseUrl: window.location.origin,
      });
    });
    expect(trackAutomationActivityLogExported).toHaveBeenCalledWith({
      backendKind: "local",
      format: "json",
    });
  });

  it("exports CSV via the activity-log export helper and tracks it", async () => {
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByTestId("activity-log-export-csv"));

    await waitFor(() => {
      expect(downloadActivityLogExport).toHaveBeenCalledWith({
        automation,
        format: "csv",
        conversationBaseUrl: window.location.origin,
      });
    });
    expect(trackAutomationActivityLogExported).toHaveBeenCalledWith({
      backendKind: "local",
      format: "csv",
    });
  });

  it("disables export when there are no runs", () => {
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: [], total: 0 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);

    renderSection();

    expect(screen.getByTestId("activity-log-export-json")).toBeDisabled();
    expect(screen.getByTestId("activity-log-export-csv")).toBeDisabled();
  });
});

describe("ActivityLogSection ?run= highlight", () => {
  function makeRuns(count: number): AutomationRun[] {
    return Array.from({ length: count }, (_, index) => ({
      ...run,
      id: `r${index + 1}`,
    }));
  }

  beforeEach(() => {
    vi.clearAllMocks();
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("marks the deep-linked run and scrolls it into view", async () => {
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: [run], total: 1 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);

    renderSection("r1");

    expect(
      screen.getByTestId("automation-run-highlight-r1"),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(window.HTMLElement.prototype.scrollIntoView).toHaveBeenCalled();
    });
  });

  it("stops auto-loading pages for a missing run id at the cap", async () => {
    vi.mocked(useAutomationRuns).mockImplementation(
      (options) =>
        ({
          data: { runs: makeRuns(options.limit ?? 20), total: 500 },
          isLoading: false,
        }) as unknown as ReturnType<typeof useAutomationRuns>,
    );

    renderSection("r-missing");

    await waitFor(() => {
      expect(useAutomationRuns).toHaveBeenCalledWith({
        id: "a1",
        limit: 100,
        offset: 0,
      });
    });

    const requestedLimits = vi
      .mocked(useAutomationRuns)
      .mock.calls.map(([options]) => options.limit ?? 0);
    expect(Math.max(...requestedLimits)).toBe(100);
    expect(window.HTMLElement.prototype.scrollIntoView).not.toHaveBeenCalled();
  });
});

describe("ActivityLogSection run grouping", () => {
  function successfulRuns(count: number): AutomationRun[] {
    return Array.from({ length: count }, (_, index) => ({
      ...run,
      id: `s${index + 1}`,
      conversation_id: `c${index + 1}`,
      started_at: `2026-01-01T09:0${index}:00Z`,
    }));
  }

  beforeEach(() => {
    vi.clearAllMocks();
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("shows the newest runs of a successful stretch and folds the rest until opened", async () => {
    // Arrange
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: successfulRuns(8), total: 8 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);
    const user = userEvent.setup();
    renderSection();
    expect(screen.getAllByTestId("automation-run-day")).toHaveLength(1);
    expect(screen.getAllByRole("link")).toHaveLength(5);

    // Act
    await user.click(screen.getByTestId("automation-run-fold"));

    // Assert
    expect(screen.getAllByRole("link")).toHaveLength(8);
    expect(screen.queryByTestId("automation-run-fold")).not.toBeInTheDocument();
  });

  it("shows every loaded run under its day once more are loaded", async () => {
    // Arrange: a run every 30 minutes, newest first, so the second page
    // reaches into the previous day.
    const runsEvery30Minutes = (count: number): AutomationRun[] =>
      Array.from({ length: count }, (_, index) => ({
        ...run,
        id: `m${index + 1}`,
        conversation_id: `c${index + 1}`,
        started_at: new Date(
          Date.UTC(2026, 0, 2, 12) - index * 30 * 60_000,
        ).toISOString(),
      }));
    vi.mocked(useAutomationRuns).mockImplementation(
      (options) =>
        ({
          data: { runs: runsEvery30Minutes(options.limit ?? 20), total: 60 },
          isLoading: false,
        }) as unknown as ReturnType<typeof useAutomationRuns>,
    );
    const user = userEvent.setup();
    renderSection();
    expect(screen.getAllByTestId("automation-run-fold")).toHaveLength(1);

    // Act
    await user.click(screen.getByTestId("automation-runs-load-more"));

    // Assert
    expect(screen.getAllByRole("link")).toHaveLength(40);
    expect(screen.queryByTestId("automation-run-fold")).not.toBeInTheDocument();
    expect(
      new Set(
        screen
          .getAllByTestId("automation-run-day")
          .map((day) => day.querySelector("h4")?.textContent),
      ).size,
    ).toBe(screen.getAllByTestId("automation-run-day").length);
  });

  it("keeps a deep-linked run visible inside a stretch it would fold", () => {
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: successfulRuns(8), total: 8 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);

    renderSection("s7");

    expect(
      screen.getByTestId("automation-run-highlight-s7"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("automation-run-fold")).not.toBeInTheDocument();
  });

  it("explains once that script runs have no conversation", () => {
    vi.mocked(useAutomationRuns).mockReturnValue({
      data: { runs: successfulRuns(2), total: 2 },
      isLoading: false,
    } as unknown as ReturnType<typeof useAutomationRuns>);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={client}>
        <ActivityLogSection automation={{ ...automation, prompt: null }} />
      </QueryClientProvider>,
    );

    expect(screen.getAllByTestId("automation-script-runs-note")).toHaveLength(
      1,
    );
  });
});
