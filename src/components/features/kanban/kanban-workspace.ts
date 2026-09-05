import { HOME_SELECTED_WORKSPACE_PATH_KEY } from "#/components/features/home/workspace-selection-form";
import type { KanbanBoardSummary } from "#/api/kanban-service/kanban-types";

export const KANBAN_SELECTED_WORKSPACE_PATH_KEY =
  "oh:kanban-selected-workspace-path";

function readStorage(key: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    const value = window.sessionStorage.getItem(key);
    return value && value.length > 0 ? value : null;
  } catch {
    return null;
  }
}

function writeStorage(key: string, path: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (path) {
      window.sessionStorage.setItem(key, path);
    } else {
      window.sessionStorage.removeItem(key);
    }
  } catch {
    // sessionStorage may be unavailable in private browsing contexts.
  }
}

export function readKanbanWorkspacePath(): string | null {
  return (
    readStorage(KANBAN_SELECTED_WORKSPACE_PATH_KEY) ??
    readStorage(HOME_SELECTED_WORKSPACE_PATH_KEY)
  );
}

export function writeKanbanWorkspacePath(path: string | null): void {
  writeStorage(KANBAN_SELECTED_WORKSPACE_PATH_KEY, path);
  writeStorage(HOME_SELECTED_WORKSPACE_PATH_KEY, path);
}

export function boardForWorkspace(
  boards: KanbanBoardSummary[],
  workspacePath: string | null,
): KanbanBoardSummary | null {
  if (!workspacePath) return null;
  return boards.find((board) => board.project_id === workspacePath) ?? null;
}
