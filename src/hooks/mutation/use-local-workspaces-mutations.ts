import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import WorkspacesService, {
  WorkspacesListResponse,
} from "#/api/workspaces-service/workspaces-service.api";
import { LOCAL_WORKSPACES_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";
import { getApiErrorMessage } from "#/utils/api-error-message";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

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

export interface RenameWorkspaceVariables {
  path: string;
  name: string;
}

/**
 * Rename a saved workspace or a workspace parent.
 *
 * `path` is the entry's identity, so only the label is rewritten — the
 * optimistic update must preserve `id`/`path` and every other field.
 */
function useRenameWorkspaceEntry(kind: "workspace" | "parent") {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");
  const isWorkspace = kind === "workspace";

  return useMutation({
    mutationFn: ({ path, name }: RenameWorkspaceVariables) =>
      isWorkspace
        ? WorkspacesService.renameWorkspace(path, name)
        : WorkspacesService.renameWorkspaceParent(path, name),
    onMutate: async ({ path, name }) => {
      await queryClient.cancelQueries({
        queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
      });
      const previous = queryClient.getQueryData<WorkspacesListResponse>(
        LOCAL_WORKSPACES_QUERY_KEYS.all,
      );

      const rename = <T extends { path: string; name: string }>(entry: T): T =>
        entry.path === path ? { ...entry, name } : entry;

      queryClient.setQueryData<WorkspacesListResponse>(
        LOCAL_WORKSPACES_QUERY_KEYS.all,
        (old) =>
          old
            ? {
                ...old,
                ...(isWorkspace
                  ? { workspaces: old.workspaces.map(rename) }
                  : { workspaceParents: old.workspaceParents.map(rename) }),
              }
            : old,
      );

      return { previous };
    },
    onError: (error, _variables, context) => {
      if (context?.previous) {
        queryClient.setQueryData(
          LOCAL_WORKSPACES_QUERY_KEYS.all,
          context.previous,
        );
      }
      displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
    },
    onSuccess: (_data, { name }) => {
      displaySuccessToast(
        t(
          isWorkspace
            ? I18nKey.HOME$WORKSPACE_RENAMED
            : I18nKey.HOME$WORKSPACE_PARENT_RENAMED,
          { name },
        ),
      );
    },
    onSettled: () =>
      queryClient.invalidateQueries({
        queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
      }),
    meta: { disableToast: true },
  });
}

export function useRenameWorkspace() {
  return useRenameWorkspaceEntry("workspace");
}

export function useRenameWorkspaceParent() {
  return useRenameWorkspaceEntry("parent");
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
