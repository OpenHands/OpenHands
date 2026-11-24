import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import * as pullRequests from "#/api/git-service/pull-request-status.api";
import {
  PULL_REQUEST_STATUS_FRESH_MS,
  usePullRequestStatuses,
} from "#/hooks/query/use-pull-request-status";
import { USER_IDLE_AFTER_MS } from "#/utils/user-activity";

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { id: "test-backend", kind: "local" } }),
}));

const PR = { owner: "acme", repo: "app", number: 7, url: "u" };

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}

/** Two places on screen showing the same pull request. */
function renderTwoSurfaces() {
  const client = new QueryClient();
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const first = renderHook(() => usePullRequestStatuses([PR]), { wrapper });
  const second = renderHook(() => usePullRequestStatuses([PR]), { wrapper });
  return () => {
    first.unmount();
    second.unmount();
  };
}

describe("usePullRequestStatuses", () => {
  let fetchStatus: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    setVisibility("visible");
    fetchStatus = vi
      .spyOn(pullRequests, "getPullRequestStatus")
      .mockResolvedValue({
        state: "open",
        draft: false,
        checks: "success",
      } as pullRequests.PullRequestStatus);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("reads a pull request once for every surface, then again only when stale", async () => {
    const unmount = renderTwoSurfaces();
    await waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));

    await act(() =>
      vi.advanceTimersByTimeAsync(PULL_REQUEST_STATUS_FRESH_MS / 2),
    );
    window.dispatchEvent(new Event("pointerdown"));
    await act(() =>
      vi.advanceTimersByTimeAsync(PULL_REQUEST_STATUS_FRESH_MS / 2 + 60_000),
    );
    window.dispatchEvent(new Event("pointerdown"));
    await act(() => vi.advanceTimersByTimeAsync(60_000));

    expect(fetchStatus).toHaveBeenCalledTimes(2);
    unmount();
  });

  it("pauses while the tab is hidden and reads stale states when it is shown", async () => {
    const unmount = renderTwoSurfaces();
    await waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));

    setVisibility("hidden");
    await act(() =>
      vi.advanceTimersByTimeAsync(3 * PULL_REQUEST_STATUS_FRESH_MS),
    );
    expect(fetchStatus).toHaveBeenCalledTimes(1);

    await act(async () => setVisibility("visible"));
    await waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(2));
    unmount();
  });

  it("pauses in a visible tab nobody touched, and reads again on return", async () => {
    const unmount = renderTwoSurfaces();
    await waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(1));

    await act(() => vi.advanceTimersByTimeAsync(USER_IDLE_AFTER_MS + 60_000));
    const readsBeforeIdle = fetchStatus.mock.calls.length;
    await act(() =>
      vi.advanceTimersByTimeAsync(3 * PULL_REQUEST_STATUS_FRESH_MS),
    );
    expect(fetchStatus).toHaveBeenCalledTimes(readsBeforeIdle);

    await act(async () => window.dispatchEvent(new Event("keydown")));
    await waitFor(() =>
      expect(fetchStatus).toHaveBeenCalledTimes(readsBeforeIdle + 1),
    );
    unmount();
  });
});
