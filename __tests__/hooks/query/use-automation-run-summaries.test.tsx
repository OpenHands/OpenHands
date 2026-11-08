import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useAutomationRunSummaries } from "#/hooks/query/use-automation-run-summaries";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
  type AutomationRunsResponse,
} from "#/types/automation";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomationRuns: vi.fn(),
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "test-backend", kind: "local" },
    orgId: null,
  }),
}));

const automations = [
  { id: "auto-1", name: "Nightly QA" },
] as unknown as Automation[];

function makeRun(status: AutomationRunStatus): AutomationRun {
  return {
    id: "run-1",
    status,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-08-01T10:00:00Z",
    completed_at:
      status === AutomationRunStatus.RUNNING ? null : "2026-08-01T10:05:00Z",
  };
}

function makeAutomation(id: string): Automation {
  return {
    id,
    name: `Automation ${id}`,
    prompt: "p",
    trigger: { type: "schedule", schedule_human: "Daily" },
    enabled: true,
    repository: "acme/repo",
    model: "daily-profile",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

const emptyRuns: AutomationRunsResponse = { runs: [], total: 0 };

function createWrapper() {
  // No retry default override here: the hook's own `retry: false` is under
  // test, and a global override would mask a regression.
  const queryClient = new QueryClient();
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// The dashboard reads its cards from this hook. Its phase — and above all the
// phase's age — is only honest if the data behind it keeps moving: a run that
// is working through its phases must not sit on the card reading "Queued ·
// 40m ago", which is exactly the stall signal the age exists to send.
describe("useAutomationRunSummaries — keeping an in-flight card current", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("refetches while a sampled run is still in flight", async () => {
    vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
      runs: [makeRun(AutomationRunStatus.RUNNING)],
      total: 1,
    });

    renderHook(() => useAutomationRunSummaries(automations), {
      wrapper: createWrapper(),
    });

    await waitFor(() =>
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(1),
    );

    await vi.advanceTimersByTimeAsync(15_000);

    await waitFor(() =>
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(2),
    );
  });

  it("stops once every sampled run has finished", async () => {
    vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
      runs: [makeRun(AutomationRunStatus.COMPLETED)],
      total: 1,
    });

    renderHook(() => useAutomationRunSummaries(automations), {
      wrapper: createWrapper(),
    });

    await waitFor(() =>
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(1),
    );

    await vi.advanceTimersByTimeAsync(60_000);

    expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(1);
  });

  it("stops when the newest run finished, despite an older stuck run", async () => {
    // A crashed dispatcher leaves an old run non-terminal forever. The card
    // renders `runs[0]` and nothing else, so polling on *any* sampled run
    // pins this automation to a permanent refetch that never changes the
    // screen — one wasted request every 15s, per automation, indefinitely.
    vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
      runs: [
        makeRun(AutomationRunStatus.COMPLETED),
        { ...makeRun(AutomationRunStatus.RUNNING), id: "run-stuck" },
      ],
      total: 2,
    });

    renderHook(() => useAutomationRunSummaries(automations), {
      wrapper: createWrapper(),
    });

    await waitFor(() =>
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(1),
    );

    await vi.advanceTimersByTimeAsync(60_000);

    expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(1);
  });
});

describe("useAutomationRunSummaries", () => {
  beforeEach(() => {
    vi.mocked(AutomationService.getAutomationRuns).mockReset();
  });

  it("keeps at most three runs requests in flight and starts the rest as responses arrive", async () => {
    // Arrange — five automations, each runs request held open by the test.
    const pending: Array<(response: AutomationRunsResponse) => void> = [];
    vi.mocked(AutomationService.getAutomationRuns).mockImplementation(
      () =>
        new Promise<AutomationRunsResponse>((resolve) => {
          pending.push(resolve);
        }),
    );
    const fanOut = ["a-1", "a-2", "a-3", "a-4", "a-5"].map(makeAutomation);

    // Act
    const { result } = renderHook(() => useAutomationRunSummaries(fanOut), {
      wrapper: createWrapper(),
    });

    // Assert — only three requests start despite five queries mounting.
    await waitFor(() => {
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(3);
    });
    expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(3);

    // One response frees a slot for exactly one queued request.
    await act(async () => {
      pending.shift()!(emptyRuns);
    });
    await waitFor(() => {
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(4);
    });

    // Draining the rest completes the whole fan-out.
    await act(async () => {
      pending.splice(0).forEach((resolve) => resolve(emptyRuns));
    });
    await waitFor(() => {
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(5);
    });
    await act(async () => {
      pending.splice(0).forEach((resolve) => resolve(emptyRuns));
    });
    await waitFor(() => {
      fanOut.forEach((automation) => {
        expect(result.current.get(automation.id)?.isLoading).toBe(false);
      });
    });
  });

  it("settles a failed summary into the error state without retrying", async () => {
    // Arrange — the runs request fails outright.
    vi.mocked(AutomationService.getAutomationRuns).mockRejectedValue(
      new Error("automation service unavailable"),
    );

    // Act
    const { result } = renderHook(
      () => useAutomationRunSummaries([makeAutomation("a-1")]),
      { wrapper: createWrapper() },
    );

    // Assert — the entry reports the error promptly (retries would keep it
    // loading for seconds) and the service was asked exactly once.
    await waitFor(() => {
      expect(result.current.get("a-1")?.isError).toBe(true);
    });
    expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(1);
    expect(result.current.get("a-1")?.summary).toBeNull();
  });
});
