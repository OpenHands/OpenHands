import { beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useLatestAutomationRuns } from "#/hooks/query/use-latest-automation-runs";
import type { Automation, AutomationRunsResponse } from "#/types/automation";

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
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("useLatestAutomationRuns", () => {
  beforeEach(() => {
    vi.mocked(AutomationService.getAutomationRuns).mockReset();
  });

  it("keeps at most three runs requests in flight across the home fan-out", async () => {
    // Arrange — five automations, each runs request held open by the test.
    const pending: Array<(response: AutomationRunsResponse) => void> = [];
    vi.mocked(AutomationService.getAutomationRuns).mockImplementation(
      () =>
        new Promise<AutomationRunsResponse>((resolve) => {
          pending.push(resolve);
        }),
    );
    const automations = ["a-1", "a-2", "a-3", "a-4", "a-5"].map(makeAutomation);

    // Act
    const { result } = renderHook(() => useLatestAutomationRuns(automations), {
      wrapper: createWrapper(),
    });

    // Assert — only three requests start; the rest follow as responses free
    // slots, until every automation's state settles.
    await waitFor(() => {
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(3);
    });
    expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(3);

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
      automations.forEach((automation) => {
        expect(result.current.get(automation.id)?.isLoading).toBe(false);
      });
    });
  });
});
