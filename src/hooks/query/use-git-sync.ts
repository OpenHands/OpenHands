import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { useTracking } from "#/hooks/use-tracking";
import type { GitSyncConfigUpdateRequest } from "#/types/git-sync";

export const GIT_SYNC_STATUS_QUERY_KEY = ["git-sync-status"] as const;

interface UseGitSyncStatusOptions {
  enabled?: boolean;
  refetchInterval?: number | false;
}

export function useGitSyncStatus(options: UseGitSyncStatusOptions = {}) {
  const { enabled = true, refetchInterval = false } = options;
  const active = useActiveBackend();
  return useQuery({
    queryKey: [...GIT_SYNC_STATUS_QUERY_KEY, active.backend.id, active.orgId],
    queryFn: () => AutomationService.getGitSyncStatus(),
    staleTime: 10 * 1000, // 10 seconds
    enabled,
    refetchInterval,
    // Only a request that never got an answer is worth repeating. A 404 means
    // the backend has no git-sync API and will answer the same forever, and an
    // answered failure has already cost the service's own timeout: a saturated
    // automation service replies 500 after a 30s connection-pool timeout, so
    // three retries plus backoff sat the page in its skeleton for about two
    // minutes before showing the error panel it could have shown at once.
    retry: (failureCount, error) =>
      getErrorStatus(error) === undefined && failureCount < 2,
    // The page turns every failure into a state of its own (unsupported
    // backend, or the error panel with Retry), so the global query toast
    // would only add raw axios text on top of it.
    meta: { disableToast: true },
  });
}

export function useUpdateGitSyncConfig() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackGitSyncConfigUpdated } = useTracking();
  return useMutation({
    mutationFn: (body: GitSyncConfigUpdateRequest) =>
      AutomationService.updateGitSyncConfig(body),
    onSuccess: async (data) => {
      const queryKey = [
        ...GIT_SYNC_STATUS_QUERY_KEY,
        active.backend.id,
        active.orgId,
      ];
      // Cancel first: a status GET that started before this save resolves
      // after it and would overwrite the response we just seeded with its own
      // pre-save snapshot.
      await queryClient.cancelQueries({ queryKey });
      queryClient.setQueryData(queryKey, data);
      trackGitSyncConfigUpdated({ backendKind: active.backend.kind });
    },
    // The form maps failures to its own message (the 409 "restart with the env
    // var set" case in particular), so the global mutation toast would stack a
    // raw one on top.
    meta: { disableToast: true },
  });
}

/**
 * Test a configuration against its remote before saving it.
 *
 * Deliberately not retried and never surfaced as a toast: the form treats a
 * check it cannot complete -- an older automation backend answering 404,
 * a network failure -- as "no opinion" and saves anyway, so a check that is
 * itself broken can never become the thing that blocks a save.
 */
export function useCheckGitSyncConfig() {
  return useMutation({
    mutationFn: (body: GitSyncConfigUpdateRequest) =>
      AutomationService.checkGitSyncConfig(body),
    retry: false,
    meta: { disableToast: true },
  });
}

export function useTriggerGitSync() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackGitSyncTriggered } = useTracking();
  return useMutation({
    mutationFn: () => AutomationService.triggerGitSync(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: GIT_SYNC_STATUS_QUERY_KEY });
      trackGitSyncTriggered({ backendKind: active.backend.kind });
    },
    // The page maps a failed trigger to its own message (503 means sync is
    // off, not that the request broke), so the global mutation toast would
    // stack a raw one on top.
    meta: { disableToast: true },
  });
}
