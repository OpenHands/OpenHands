import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  useAutomations,
  useDispatchAutomation,
  useDeleteAutomation,
  useToggleAutomation,
  useUpdateAutomation,
} from "#/hooks/query/use-automations";
import {
  useAutomationDetail,
  useAutomationRuns,
} from "#/hooks/query/use-automation-detail";
import type { Backend } from "#/api/backend-registry/types";
import { AutomationRunStatus } from "#/types/automation";
import type {
  Automation,
  AutomationRun,
  AutomationsResponse,
  AutomationRunsResponse,
} from "#/types/automation";
import * as telemetry from "#/services/telemetry";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomations: vi.fn(),
    getAutomation: vi.fn(),
    getAutomationRuns: vi.fn(),
    dispatchAutomation: vi.fn(),
    deleteAutomation: vi.fn(),
    updateAutomation: vi.fn(),
    toggleAutomation: vi.fn(),
  },
}));

let captureMock: ReturnType<typeof vi.spyOn>;

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: { user_consents_to_analytics: true } }),
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const automation: Automation = {
  id: "auto-1",
  name: "Test",
  prompt: "p",
  trigger: { type: "schedule", schedule_human: "Daily" },
  enabled: true,
  repository: "acme/repo",
  model: "daily-profile",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const automationRun: AutomationRun = {
  id: "run-1",
  status: AutomationRunStatus.PENDING,
  conversation_id: null,
  bash_command_id: null,
  error_detail: null,
  started_at: "2026-01-02T00:00:00Z",
  completed_at: null,
};

const listResponse: AutomationsResponse = {
  automations: [automation],
  total: 1,
};

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

beforeEach(() => {
  captureMock = vi.spyOn(telemetry, "trackEvent").mockResolvedValue(undefined);
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(AutomationService.getAutomations).mockReset();
  vi.mocked(AutomationService.getAutomation).mockReset();
  vi.mocked(AutomationService.getAutomationRuns).mockReset();
  vi.mocked(AutomationService.dispatchAutomation).mockReset();
  vi.mocked(AutomationService.dispatchAutomation).mockResolvedValue(
    automationRun,
  );
  vi.mocked(AutomationService.deleteAutomation).mockReset();
  vi.mocked(AutomationService.updateAutomation).mockReset();
  vi.mocked(AutomationService.toggleAutomation).mockReset();
  vi.mocked(AutomationService.deleteAutomation).mockResolvedValue(undefined);
  vi.mocked(AutomationService.updateAutomation).mockResolvedValue(automation);
  vi.mocked(AutomationService.toggleAutomation).mockResolvedValue(automation);

  vi.mocked(AutomationService.getAutomations).mockResolvedValue(listResponse);
  vi.mocked(AutomationService.getAutomation).mockResolvedValue(automation);
  setRegisteredBackends([localBackend, cloudBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  captureMock.mockRestore();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("automation hooks — backend switch", () => {
  it("useAutomations refetches when the active backend changes", async () => {
    // Arrange — mount under the local backend; capture the initial fetch.
    const { result } = renderHook(
      () => useAutomations({ limit: 50, offset: 0 }),
      { wrapper: makeWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1);

    // Act — flip the active backend to a cloud one.
    setActiveSelection({ backendId: cloudBackend.id });

    // Assert — react-query treats the new (backend, org) as a brand-new
    // query (the key includes active.backend.id + active.orgId), so a
    // second fetch fires automatically without any explicit invalidate.
    await waitFor(() => {
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(2);
    });
  });

  it("useAutomationDetail refetches when the active backend changes", async () => {
    const { result } = renderHook(() => useAutomationDetail({ id: "auto-1" }), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(AutomationService.getAutomation).toHaveBeenCalledTimes(1);

    setActiveSelection({ backendId: cloudBackend.id });

    await waitFor(() => {
      expect(AutomationService.getAutomation).toHaveBeenCalledTimes(2);
    });
  });

  it("useDispatchAutomation dispatches the selected automation", async () => {
    const { result } = renderHook(() => useDispatchAutomation(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.mutateAsync("auto-1");
    });

    expect(AutomationService.dispatchAutomation).toHaveBeenCalledWith("auto-1");
  });
});

describe("useAutomationRuns — polling", () => {
  const pendingRun: AutomationRun = {
    id: "run-pending",
    status: AutomationRunStatus.PENDING,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-01-02T00:00:00Z",
    completed_at: null,
  };
  const completedRun: AutomationRun = {
    id: "run-pending",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: "conv-1",
    bash_command_id: "cmd-1",
    error_detail: null,
    started_at: "2026-01-02T00:00:00Z",
    completed_at: "2026-01-02T00:00:30Z",
  };

  it(
    "re-fetches while a run is non-terminal, and stops once all runs are terminal",
    async () => {
      // Arrange: first fetch returns a PENDING run (polling should engage);
      // subsequent fetches return a COMPLETED run (polling should then stop).
      const pendingResponse: AutomationRunsResponse = {
        runs: [pendingRun],
        total: 1,
      };
      const completedResponse: AutomationRunsResponse = {
        runs: [completedRun],
        total: 1,
      };
      vi.mocked(AutomationService.getAutomationRuns)
        .mockResolvedValueOnce(pendingResponse)
        .mockResolvedValue(completedResponse);

      // Act
      renderHook(
        () => useAutomationRuns({ id: "auto-1", limit: 20, offset: 0 }),
        { wrapper: makeWrapper() },
      );

      // Assert: the initial fetch fires once.
      await waitFor(() => {
        expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(1);
      });

      // The cached data still contains a PENDING run, so refetchInterval
      // engages and a second fetch arrives within the poll window.
      await waitFor(
        () => {
          expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(2);
        },
        { timeout: 5000 },
      );

      // The second fetch returned a COMPLETED run, so polling should stop.
      // Give the would-be next poll window plenty of slack and assert no
      // further calls happen.
      await new Promise((resolve) => {
        setTimeout(resolve, 4000);
      });
      expect(AutomationService.getAutomationRuns).toHaveBeenCalledTimes(2);
    },
    15000,
  );
});

describe("automation mutation hooks — analytics tracking", () => {
  it("captures automation_executed after a successful dispatch", async () => {
    const { result } = renderHook(() => useDispatchAutomation(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.mutateAsync("auto-1");
    });

    await waitFor(() => {
      expect(captureMock).toHaveBeenCalledWith(
        "automation_executed",
        expect.objectContaining({ backend_kind: "local" }),
      );
    });
  });

  it("captures automation_deleted after a successful delete", async () => {
    const { result } = renderHook(() => useDeleteAutomation(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.mutateAsync("auto-1");
    });

    await waitFor(() => {
      expect(captureMock).toHaveBeenCalledWith(
        "automation_deleted",
        expect.objectContaining({ backend_kind: "local" }),
      );
    });
  });

  it("captures automation_edited after a successful update", async () => {
    const { result } = renderHook(() => useUpdateAutomation(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.mutateAsync({
        id: "auto-1",
        body: { name: "Renamed" },
      });
    });

    await waitFor(() => {
      expect(captureMock).toHaveBeenCalledWith(
        "automation_edited",
        expect.objectContaining({ backend_kind: "local" }),
      );
    });
  });

  it("captures automation_disable_button when an automation is disabled", async () => {
    const { result } = renderHook(() => useToggleAutomation(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.mutateAsync({ id: "auto-1", enabled: false });
    });

    await waitFor(() => {
      expect(captureMock).toHaveBeenCalledWith(
        "automation_disable_button",
        expect.objectContaining({ backend_kind: "local" }),
      );
    });
  });

  it("does not capture automation_disable_button when an automation is enabled", async () => {
    const { result } = renderHook(() => useToggleAutomation(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.mutateAsync({ id: "auto-1", enabled: true });
    });

    expect(captureMock).not.toHaveBeenCalledWith(
      "automation_disable_button",
      expect.anything(),
    );
  });
});
