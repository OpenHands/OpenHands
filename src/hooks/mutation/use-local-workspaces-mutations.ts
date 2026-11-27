import { useMutation, useQueryClient } from "@tanstack/react-query";

import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { LOCAL_WORKSPACES_QUERY_KEYS } from "#/hooks/query/query-keys";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

export function useAddWorkspaces() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (items: LocalWorkspace[]) =>
      WorkspacesService.addWorkspaces(items),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
      }),
  });
}

export function useRemoveWorkspace() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (path: string) => WorkspacesService.removeWorkspace(path),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
      }),
  });
}

export function useAddWorkspaceParents() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (items: LocalWorkspaceParent[]) =>
      WorkspacesService.addWorkspaceParents(items),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
      }),
  });
}

export function useRemoveWorkspaceParent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (path: string) => WorkspacesService.removeWorkspaceParent(path),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
      }),
  });
}
