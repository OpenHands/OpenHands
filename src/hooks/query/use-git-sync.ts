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
    // An automation backend without the git-sync API answers 404 forever --
    // retrying only delays the "unsupported backend" state the page renders
    // for it.
    retry: (failureCount, error) =>
      getErrorStatus(error) !== 404 && failureCount < 3,
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
