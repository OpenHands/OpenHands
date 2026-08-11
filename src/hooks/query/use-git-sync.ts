import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
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
  });
}

export function useUpdateGitSyncConfig() {
  const queryClient = useQueryClient();
  const active = useActiveBackend();
  const { trackGitSyncConfigUpdated } = useTracking();
  return useMutation({
    mutationFn: (body: GitSyncConfigUpdateRequest) =>
      AutomationService.updateGitSyncConfig(body),
    onSuccess: (data) => {
      queryClient.setQueryData(
        [...GIT_SYNC_STATUS_QUERY_KEY, active.backend.id, active.orgId],
        data,
      );
      trackGitSyncConfigUpdated({ backendKind: active.backend.kind });
    },
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
  });
}
