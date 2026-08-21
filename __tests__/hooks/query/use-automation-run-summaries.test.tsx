import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { useAutomationRunSummaries } from "#/hooks/query/use-automation-run-summaries";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
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

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
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
});
