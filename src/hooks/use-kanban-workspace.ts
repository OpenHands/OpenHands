import React from "react";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useResolvedWorkspaces } from "#/hooks/query/use-resolved-workspaces";
import type { LocalWorkspace } from "#/types/workspace";
import {
  readKanbanWorkspacePath,
  subscribeKanbanWorkspacePath,
  writeKanbanWorkspacePath,
} from "#/components/features/kanban/kanban-workspace";

export function useKanbanWorkspace() {
  const { data: workspacesData, error: workspacesError } = useLocalWorkspaces();
  const workspaceParents = workspacesData?.workspaceParents ?? [];
  const {
    workspaces,
    parents,
    isLoading,
    isError,
    error: resolvedError,
  } = useResolvedWorkspaces();
  const [path, setPath] = React.useState<string | null>(
    readKanbanWorkspacePath,
  );
  const didInit = React.useRef(false);

  const selected =
    workspaces.find((workspace) => workspace.path === path) ?? null;

  const setSelected = React.useCallback((workspace: LocalWorkspace | null) => {
    setPath(workspace?.path ?? null);
    writeKanbanWorkspacePath(workspace?.path ?? null);
  }, []);

  React.useEffect(
    () => subscribeKanbanWorkspacePath((next) => setPath(next)),
    [],
  );

  React.useEffect(() => {
    if (isLoading) return;
    if (!didInit.current) {
      didInit.current = true;
      if (path && workspaces.some((workspace) => workspace.path === path)) {
        return;
      }
      const fallback = workspaces[0] ?? null;
      if (fallback) setSelected(fallback);
      return;
    }
    if (path && !workspaces.some((workspace) => workspace.path === path)) {
      setSelected(workspaces[0] ?? null);
    }
  }, [isLoading, path, workspaces, setSelected]);

  return {
    workspaces,
    parents,
    workspaceParents,
    selected,
    setSelected,
    isLoading,
    isError,
    listError: workspacesError ?? resolvedError,
  };
}
