import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import {
  GIT_SYNC_STATUS_QUERY_KEY,
  useGitSyncStatus,
  useTriggerGitSync,
  useUpdateGitSyncConfig,
} from "#/hooks/query/use-git-sync";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { GitSyncStatus } from "#/types/git-sync";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getGitSyncStatus: vi.fn(),
    updateGitSyncConfig: vi.fn(),
    triggerGitSync: vi.fn(),
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "test-backend", kind: "local" },
    orgId: null,
  }),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackGitSyncConfigUpdated: vi.fn(),
    trackGitSyncTriggered: vi.fn(),
  }),
}));

const STATUS_KEY = [...GIT_SYNC_STATUS_QUERY_KEY, "test-backend", null];

const baseStatus: GitSyncStatus = {
  enabled: true,
  repo_url: "https://example.com/org/repo.git",
  branch: "main",
  path: "automations",
  encryption_enabled: false,
  interval_seconds: 0,
  last_synced_commit: null,
  last_synced_at: null,
  last_error: null,
  last_error_at: null,
  dirty_count: 0,
};

function createHarness() {
  const queryClient = new QueryClient({
    // `retry` is the default for queries that don't set their own; the hooks
    // under test do, so `retryDelay` is what matters here -- without it the
    // real exponential backoff would make a retry test wait a second.
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { queryClient, wrapper };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("useUpdateGitSyncConfig", () => {
  it("does not let a status fetch started before the save overwrite it", async () => {
    // React Query resolves fetches in completion order, not request order: a
    // GET in flight when the save lands would otherwise replace the saved
    // config with its own pre-save snapshot.
    let resolveStaleFetch: (value: GitSyncStatus) => void = () => {};
    vi.mocked(AutomationService.getGitSyncStatus).mockReturnValue(
      new Promise<GitSyncStatus>((resolve) => {
        resolveStaleFetch = resolve;
      }),
    );
    const saved: GitSyncStatus = { ...baseStatus, branch: "develop" };
    vi.mocked(AutomationService.updateGitSyncConfig).mockResolvedValue(saved);

    const { queryClient, wrapper } = createHarness();
    const { result } = renderHook(
      () => ({
        status: useGitSyncStatus(),
        update: useUpdateGitSyncConfig(),
      }),
      { wrapper },
    );

    await act(async () => {
      await result.current.update.mutateAsync({ branch: "develop" });
    });
    expect(queryClient.getQueryData(STATUS_KEY)).toEqual(saved);

    // The pre-save GET now comes back with the old branch.
    await act(async () => {
      resolveStaleFetch(baseStatus);
      await Promise.resolve();
    });

    expect(queryClient.getQueryData(STATUS_KEY)).toEqual(saved);
  });

  it("leaves error reporting to the caller", async () => {
    // The page maps 409/503 to specific guidance; the global mutation toast
    // would stack the raw axios message on top of it.
    vi.mocked(AutomationService.updateGitSyncConfig).mockRejectedValue(
      new Error("boom"),
    );
    const { queryClient, wrapper } = createHarness();
    const { result } = renderHook(() => useUpdateGitSyncConfig(), { wrapper });

    act(() => {
      result.current.mutate({ branch: "develop" });
    });
    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(queryClient.getMutationCache().getAll()[0].options.meta).toEqual({
      disableToast: true,
    });
  });
});

describe("useTriggerGitSync", () => {
  it("leaves error reporting to the caller", async () => {
    vi.mocked(AutomationService.triggerGitSync).mockRejectedValue(
      new Error("boom"),
    );
    const { queryClient, wrapper } = createHarness();
    const { result } = renderHook(() => useTriggerGitSync(), { wrapper });

    act(() => {
      result.current.mutate();
    });
    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(queryClient.getMutationCache().getAll()[0].options.meta).toEqual({
      disableToast: true,
    });
  });
});

describe("useGitSyncStatus", () => {
  it("does not retry a backend that has no git-sync API", async () => {
    vi.mocked(AutomationService.getGitSyncStatus).mockRejectedValue({
      status: 404,
    });
    const { wrapper } = createHarness();

    const { result } = renderHook(() => useGitSyncStatus(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(AutomationService.getGitSyncStatus).toHaveBeenCalledTimes(1);
  });

  // Regression: this retried anything that wasn't a 404. The automation
  // service answers 500 only after its own 30s connection-pool timeout, so
  // three retries plus backoff held the page's skeleton for about two minutes
  // instead of showing the error panel with its Retry button straight away.
  it("does not retry an answered failure", async () => {
    vi.mocked(AutomationService.getGitSyncStatus).mockRejectedValue({
      status: 500,
    });
    const { wrapper } = createHarness();

    const { result } = renderHook(() => useGitSyncStatus(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(AutomationService.getGitSyncStatus).toHaveBeenCalledTimes(1);
  });

  it("retries a request that never got an answer", async () => {
    // No status: the request never reached the service, so repeating it is
    // cheap and can actually succeed.
    vi.mocked(AutomationService.getGitSyncStatus)
      .mockRejectedValueOnce(new Error("network error"))
      .mockResolvedValue(baseStatus);
    const { wrapper } = createHarness();

    const { result } = renderHook(() => useGitSyncStatus(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(AutomationService.getGitSyncStatus).toHaveBeenCalledTimes(2);
    expect(result.current.data).toEqual(baseStatus);
  });
});
