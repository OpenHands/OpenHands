import { useSyncExternalStore } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getSnapshot,
  subscribeActiveBackend,
} from "#/api/backend-registry/active-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { getSettingsQueryFn } from "#/hooks/query/use-settings";
import {
  normalizeFileDiscovery,
  type WorkspaceFileDiscovery,
} from "#/utils/workspace-file-discovery";

// @spec WFD-002 — Workspace-scoped server persistence
export function useWorkspaceFileDiscovery(workingDir?: string) {
  const { active } = useSyncExternalStore(
    subscribeActiveBackend,
    getSnapshot,
    getSnapshot,
  );
  const isLocal = active.backend.kind === "local";
  const client = useQueryClient();
  const query = useQuery({
    queryKey: [
      ...SETTINGS_QUERY_KEYS.personal(),
      active.backend.id,
      active.orgId,
    ],
    queryFn: () => getSettingsQueryFn(),
    enabled: isLocal && !!workingDir,
    staleTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });
  const backendId = active.backend.id;
  const mutation = useMutation({
    mutationFn: async (options: WorkspaceFileDiscovery) => {
      // A dialog opened on one backend must never write into another.
      if (
        !workingDir ||
        getSnapshot().active.backend.id !== backendId ||
        !isLocal
      ) {
        throw new Error("Workspace backend changed");
      }
      await SettingsService.saveSettings({
        workspace_file_discovery: { [workingDir]: options },
      });
    },
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
    },
    meta: { disableToast: true },
  });
  return {
    options: normalizeFileDiscovery(
      query.data?.workspace_file_discovery?.[workingDir ?? ""],
    ),
    isLoading: isLocal && !!workingDir && query.isLoading,
    canConfigure: isLocal && !!workingDir && query.isSuccess,
    scopeKey: `${backendId}:${workingDir ?? ""}`,
    save: mutation.mutateAsync,
    isSaving: mutation.isPending,
  };
}
