import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import { HttpError } from "@openhands/typescript-client";

import { I18nKey } from "#/i18n/declaration";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import AutomationGitSync from "#/routes/automation-git-sync";
import type { Backend } from "#/api/backend-registry/types";
import type { GitSyncStatus } from "#/types/git-sync";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getGitSyncStatus: vi.fn(),
    updateGitSyncConfig: vi.fn(),
    triggerGitSync: vi.fn(),
    checkHealth: vi.fn(),
  },
}));

const displayErrorToast = vi.fn();
const displaySuccessToast = vi.fn();

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (message: string) => displayErrorToast(message),
  displaySuccessToast: (message: string) => displaySuccessToast(message),
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const status: GitSyncStatus = {
  enabled: true,
  repo_url: "https://example.com/org/repo.git",
  branch: "main",
  path: "automations",
  encryption_enabled: false,
  interval_seconds: 0,
  last_synced_commit: "abc1234",
  last_synced_at: "2026-08-10T00:00:00Z",
  last_error: null,
  last_error_at: null,
  dirty_count: 0,
};

function renderGitSync() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const view = render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={["/automations/git-sync"]}>
          <Routes>
            <Route
              path="/automations/git-sync"
              element={<AutomationGitSync />}
            />
          </Routes>
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return { ...view, queryClient };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(AutomationService.checkHealth).mockReset();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
  vi.mocked(AutomationService.getGitSyncStatus).mockReset();
  vi.mocked(AutomationService.getGitSyncStatus).mockResolvedValue(status);
  vi.mocked(AutomationService.triggerGitSync).mockReset();
  vi.mocked(AutomationService.triggerGitSync).mockResolvedValue({
    triggered: true,
  });
  displayErrorToast.mockReset();
  displaySuccessToast.mockReset();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  vi.useRealTimers();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("AutomationGitSync — backend without the git-sync API", () => {
  it("names the missing API instead of showing the generic error panel", async () => {
    // Every released automation version below the one that ships the router
    // answers 404 here, which used to settle on "something went wrong".
    vi.mocked(AutomationService.getGitSyncStatus).mockRejectedValue(
      new HttpError(404, "Not Found", { detail: "Not Found" }),
    );

    renderGitSync();

    expect(
      await screen.findByText(I18nKey.AUTOMATIONS$GIT_SYNC$UNSUPPORTED_TITLE),
    ).toBeInTheDocument();
    // A permanent 404 is not worth retrying.
    expect(AutomationService.getGitSyncStatus).toHaveBeenCalledTimes(1);
  });
});

describe("AutomationGitSync — failed background refetch", () => {
  it("keeps the loaded page instead of replacing it with the error panel", async () => {
    vi.useFakeTimers();
    renderGitSync();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByTestId("git-sync-repo-url-input")).toBeInTheDocument();

    // Break the endpoint the way an automation backend restart would, then
    // let the post-trigger poll run into it.
    vi.mocked(AutomationService.getGitSyncStatus).mockRejectedValue(
      new HttpError(502, "Bad Gateway", { detail: "upstream restarting" }),
    );
    await act(async () => {
      fireEvent.click(screen.getByTestId("git-sync-now-button"));
      // Several 3s polls plus their retries and backoff.
      await vi.advanceTimersByTimeAsync(20_000);
    });

    // The config form is still mounted, so nothing half-typed in it is lost.
    expect(screen.getByTestId("git-sync-repo-url-input")).toBeInTheDocument();
    expect(screen.queryByText(I18nKey.ERROR$GENERIC)).not.toBeInTheDocument();
  });
});

describe("AutomationGitSync — sync progress", () => {
  /** Load the page with fake timers already running. */
  async function loadPage() {
    vi.useFakeTimers();
    renderGitSync();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    return screen.getByTestId("git-sync-now-button");
  }

  const activityRow = () => screen.queryByTestId("git-sync-activity-row");

  it("reports the running cycle in the page rather than with a toast", async () => {
    // The trigger returns as soon as the cycle is scheduled, so a toast can
    // only ever say "started" and never how it ended.
    const syncNow = await loadPage();

    await act(async () => {
      fireEvent.click(syncNow);
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(activityRow()).toHaveAttribute("data-state", "running");
    expect(displaySuccessToast).not.toHaveBeenCalled();
    expect(syncNow).toHaveTextContent(I18nKey.AUTOMATIONS$GIT_SYNC$SYNCING);
    expect(syncNow).toBeDisabled();
  });

  it("resolves as soon as the cycle lands, without waiting out the window", async () => {
    const syncNow = await loadPage();
    await act(async () => {
      fireEvent.click(syncNow);
      await vi.advanceTimersByTimeAsync(0);
    });

    // The backend finishes the cycle: a newer success than the one the page
    // started from.
    vi.mocked(AutomationService.getGitSyncStatus).mockResolvedValue({
      ...status,
      last_synced_at: "2026-08-11T00:00:00Z",
      last_synced_commit: "def5678",
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000);
    });

    expect(activityRow()).toHaveAttribute("data-state", "succeeded");
    // Released well inside the 30s window the page used to sit out.
    expect(syncNow).not.toBeDisabled();
  });

  it("reports a cycle that failed", async () => {
    const syncNow = await loadPage();
    await act(async () => {
      fireEvent.click(syncNow);
      await vi.advanceTimersByTimeAsync(0);
    });

    vi.mocked(AutomationService.getGitSyncStatus).mockResolvedValue({
      ...status,
      last_error: "fatal: Authentication failed",
      last_error_at: "2026-08-11T00:00:00Z",
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000);
    });

    expect(activityRow()).toHaveAttribute("data-state", "failed");
    expect(screen.getByTestId("git-sync-error-banner")).toHaveTextContent(
      "fatal: Authentication failed",
    );
  });

  it("follows a cycle the backend started on its own", async () => {
    // An interval-driven sync, or one triggered from another tab: the page
    // never asked for it, so only `sync_in_progress` reveals it.
    vi.mocked(AutomationService.getGitSyncStatus).mockResolvedValue({
      ...status,
      sync_in_progress: true,
      sync_started_at: "2026-08-10T00:00:05Z",
    });

    const syncNow = await loadPage();

    expect(activityRow()).toHaveAttribute("data-state", "running");
    expect(syncNow).toBeDisabled();
    expect(AutomationService.triggerGitSync).not.toHaveBeenCalled();
  });

  it("gives up on a backend that never reports the cycle", async () => {
    // Fallback for an automation backend without `sync_in_progress`: the page
    // can only wait a bounded time for an outcome that may never come.
    const syncNow = await loadPage();
    await act(async () => {
      fireEvent.click(syncNow);
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(activityRow()).toHaveAttribute("data-state", "running");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(31_000);
    });

    expect(activityRow()).not.toBeInTheDocument();
    expect(syncNow).not.toBeDisabled();
  });

  it("keeps reporting the trigger's own failure with a toast", async () => {
    // A rejected request never becomes a cycle, so there is no progress to
    // show for it -- 503 means sync is switched off.
    vi.mocked(AutomationService.triggerGitSync).mockRejectedValue({
      status: 503,
    });
    const syncNow = await loadPage();

    await act(async () => {
      fireEvent.click(syncNow);
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(displayErrorToast).toHaveBeenCalledWith(
      I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_DISABLED_ERROR,
    );
    expect(activityRow()).not.toBeInTheDocument();
  });
});
