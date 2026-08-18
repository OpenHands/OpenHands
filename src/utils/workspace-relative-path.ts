import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";

function normalizeWorkspaceRoot(root: string): string {
  const trimmed = root.trim();
  const withLeading = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withLeading.replace(/\/+$/, "");
}

/**
 * Convert an agent file path into a workspace-relative path for the Files
 * drawer / workspace file APIs.
 *
 * File-editor events typically carry absolute sandbox paths
 * (e.g. `/workspace/project/report.md`) while `selectedPath` and
 * `useWorkspaceFileContent` expect paths relative to the conversation
 * working directory (e.g. `report.md`).
 */
export function toWorkspaceRelativePath(
  path: string,
  workingDir?: string | null,
): string {
  const trimmedPath = path.trim();
  if (!trimmedPath) {
    return trimmedPath;
  }

  if (!trimmedPath.startsWith("/")) {
    return trimmedPath.replace(/^\.\//, "");
  }

  const roots = [workingDir, DEFAULT_WORKING_DIR].filter(
    (value): value is string => Boolean(value?.trim()),
  );

  for (const root of roots) {
    const normalizedRoot = normalizeWorkspaceRoot(root);
    if (trimmedPath === normalizedRoot) {
      return "";
    }
    if (trimmedPath.startsWith(`${normalizedRoot}/`)) {
      return trimmedPath.slice(normalizedRoot.length + 1);
    }
  }

  return trimmedPath.replace(/^\/+/, "");
}
