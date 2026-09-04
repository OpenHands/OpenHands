export const PROJECTS_PATH = "/projects";
export const PROJECTS_API_PATH = "/api/projects";
export const SESSION_API_KEY_HEADER = "X-Session-API-Key";
export const DEFAULT_PROJECT_BRANCH = "main";

export const PROJECT_STATUSES = ["active", "idle", "error"] as const;
export const WORKTREE_STATUSES = [
  "idle",
  "working",
  "reviewing",
  "ci",
  "merged",
  "error",
] as const;

export function projectDetailPath(projectId: string): string {
  return `${PROJECTS_PATH}/${encodeURIComponent(projectId)}`;
}

export function projectIdFromPath(currentPath: string): string | null {
  if (!currentPath.startsWith(`${PROJECTS_PATH}/`)) {
    return null;
  }
  const rest = currentPath.slice(PROJECTS_PATH.length + 1);
  if (!rest || rest.includes("/")) {
    return null;
  }
  return decodeURIComponent(rest);
}
