import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";
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
import { GitSyncErrorBanner } from "#/components/features/automations/git-sync/git-sync-error-banner";
import { GitSyncOverviewSection } from "#/components/features/automations/git-sync/git-sync-overview-section";
import { GitSyncConfigForm } from "#/components/features/automations/git-sync/git-sync-config-form";

// How long to keep polling the status endpoint after a manual trigger so
// the fire-and-forget backend cycle's eventual result (new commit, dirty
// count, or error) shows up without a page refresh.
const POLL_WINDOW_MS = 30_000;
const POLL_INTERVAL_MS = 3_000;

export default function AutomationGitSync() {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const canManage = useHasPermission("manage_automations");
  const [isPolling, setIsPolling] = useState(false);

  useEffect(() => {
    if (!isPolling) return undefined;
    const timer = setTimeout(() => setIsPolling(false), POLL_WINDOW_MS);
    return () => clearTimeout(timer);
  }, [isPolling]);

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
    isError,
    refetch,
  } = useGitSyncStatus({
    enabled: isBackendHealthy && isLocalBackend,
    refetchInterval: isPolling ? POLL_INTERVAL_MS : false,
  });

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

  if (isError || !status) {
    return (
      <div className="min-h-full">
        <div className="p-6 max-w-4xl mx-auto">
          <ErrorState onRetry={() => refetch()} />
        </div>
      </div>
    );
  }

  const handleSyncNow = () => {
    triggerMutation.mutate(undefined, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_TRIGGERED));
        setIsPolling(true);
      },
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
            isSyncing={triggerMutation.isPending}
            canManage={canManage}
          />
          <GitSyncConfigForm status={status} canManage={canManage} />
        </div>
      </div>
    </div>
  );
}
