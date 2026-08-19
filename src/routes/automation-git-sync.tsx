import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { getErrorStatus } from "#/hooks/query/use-settings";
import {
  useGitSyncStatus,
  useTriggerGitSync,
} from "#/hooks/query/use-git-sync";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useHasPermission } from "#/hooks/use-has-permission";
import { BackLink } from "#/components/features/automations/detail/back-link";
import { ErrorState } from "#/components/features/automations/error-state";
import { BackendNotConfigured } from "#/components/features/automations/backend-not-configured";
import { GitSyncSkeleton } from "#/components/features/automations/git-sync/git-sync-skeleton";
import { GitSyncNotLocalState } from "#/components/features/automations/git-sync/git-sync-not-local-state";
import { GitSyncUnsupportedState } from "#/components/features/automations/git-sync/git-sync-unsupported-state";
import { GitSyncErrorBanner } from "#/components/features/automations/git-sync/git-sync-error-banner";
import { GitSyncOverviewSection } from "#/components/features/automations/git-sync/git-sync-overview-section";
import { GitSyncConfigForm } from "#/components/features/automations/git-sync/git-sync-config-form";
import type { GitSyncStatus } from "#/types/git-sync";

// While a cycle is running the status is followed closely, so its result
// (new commit, dirty count, or error) lands without a page refresh; the idle
// cadence exists to notice a cycle the backend's own interval started.
const POLL_INTERVAL_MS = 3_000;
const IDLE_POLL_INTERVAL_MS = 15_000;
// How long to keep following a cycle we triggered but have never seen the
// backend report as running -- the fallback for an automation backend that
// predates `sync_in_progress`.
const POLL_WINDOW_MS = 30_000;

type SyncActivity =
  | { state: "idle" | "succeeded" | "failed" }
  | {
      state: "running";
      startedAt: string;
      // What the last outcome was when this cycle began: the cycle has landed
      // once the backend reports a newer success or failure than these.
      lastSyncedAt: string | null;
      lastErrorAt: string | null;
    };

const runningSince = (status: GitSyncStatus): SyncActivity => ({
  state: "running",
  startedAt: status.sync_started_at ?? new Date().toISOString(),
  lastSyncedAt: status.last_synced_at,
  lastErrorAt: status.last_error_at,
});

export default function AutomationGitSync() {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const canManage = useHasPermission("manage_automations");
  const [activity, setActivity] = useState<SyncActivity>({ state: "idle" });
  const isRunning = activity.state === "running";

  const {
    data: healthData,
    isLoading: isHealthLoading,
    refetch: refetchHealth,
  } = useAutomationHealth();
  const isBackendHealthy = healthData?.status === "ok";

  const isLocalBackend = active.backend.kind === "local";

  const {
    data: status,
    isLoading,
    error,
    refetch,
  } = useGitSyncStatus({
    enabled: isBackendHealthy && isLocalBackend,
    refetchInterval: isRunning ? POLL_INTERVAL_MS : IDLE_POLL_INTERVAL_MS,
  });

  // Follow a cycle this page did not trigger -- the periodic loop's, or
  // another operator's -- so it shows up here too.
  useEffect(() => {
    if (!status?.sync_in_progress) return;
    setActivity((current) =>
      current.state === "running" ? current : runningSince(status),
    );
  }, [status]);

  // A cycle is done once the backend reports an outcome newer than the one it
  // started from. That releases the page as soon as the sync really ends,
  // rather than at the end of a fixed window.
  useEffect(() => {
    if (activity.state !== "running" || !status) return;
    if (status.last_error_at && status.last_error_at !== activity.lastErrorAt) {
      setActivity({ state: "failed" });
    } else if (
      status.last_synced_at &&
      status.last_synced_at !== activity.lastSyncedAt
    ) {
      setActivity({ state: "succeeded" });
    }
  }, [status, activity]);

  // Give up on a cycle we never saw the backend confirm. A backend that does
  // report `sync_in_progress` immediately re-enters the running state above,
  // so this only bounds the blind case.
  useEffect(() => {
    if (!isRunning) return undefined;
    const timer = setTimeout(
      () => setActivity({ state: "idle" }),
      POLL_WINDOW_MS,
    );
    return () => clearTimeout(timer);
  }, [isRunning, activity]);

  const triggerMutation = useTriggerGitSync();

  if (!isLocalBackend) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <GitSyncNotLocalState />
        </div>
      </div>
    );
  }

  if (isHealthLoading) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <GitSyncSkeleton />
        </div>
      </div>
    );
  }

  if (!isBackendHealthy) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <BackendNotConfigured onRetry={refetchHealth} />
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <GitSyncSkeleton />
        </div>
      </div>
    );
  }

  // A missing status replaces the page; a failure with one still cached does
  // not. `isError` alone is true for a failed background poll too, and this
  // page polls every few seconds after a sync -- unmounting on one of those
  // would throw away whatever is half-typed in the config form below.
  if (!status) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          {getErrorStatus(error) === 404 ? (
            <GitSyncUnsupportedState />
          ) : (
            <ErrorState onRetry={() => refetch()} />
          )}
        </div>
      </div>
    );
  }

  const handleSyncNow = () => {
    triggerMutation.mutate(undefined, {
      // No success toast: the trigger only means the cycle was scheduled, and
      // a toast that fires and disappears can't report how it ends. The
      // activity row in the card follows it through to its outcome instead --
      // including when `triggered` is false, which means a cycle was already
      // running and is the one to follow.
      onSuccess: () => setActivity(runningSince(status)),
      onError: (error) => {
        const errorStatus = getErrorStatus(error);
        displayErrorToast(
          errorStatus === 503
            ? t(I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_DISABLED_ERROR)
            : getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)),
        );
      },
    });
  };

  return (
    <div className="min-h-full">
      <div className="p-6 max-w-4xl mx-auto">
        <div className="flex flex-col gap-4">
          <BackLink />
          <div>
            <h1 className="text-xl font-semibold text-content">
              {t(I18nKey.AUTOMATIONS$GIT_SYNC$TITLE)}
            </h1>
            <p className="mt-1 text-sm text-muted">
              {t(I18nKey.AUTOMATIONS$GIT_SYNC$SUBTITLE)}
            </p>
          </div>
          {status.last_error && (
            <GitSyncErrorBanner
              error={status.last_error}
              errorAt={status.last_error_at}
            />
          )}
          <GitSyncOverviewSection
            status={status}
            onSyncNow={handleSyncNow}
            // The POST returns as soon as the cycle is scheduled, so the
            // activity state is what tracks the sync actually running.
            isSyncing={triggerMutation.isPending || isRunning}
            syncActivity={activity.state}
            syncStartedAt={isRunning ? activity.startedAt : null}
            canManage={canManage}
          />
          <GitSyncConfigForm
            status={status}
            canManage={canManage}
            onSyncNow={handleSyncNow}
          />
        </div>
      </div>
    </div>
  );
}
