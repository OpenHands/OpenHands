import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";

export function getGitPath(
  selectedRepository: string | null | undefined,
  workingDir?: string | null,
): string {
  const normalizedWorkingDir = workingDir?.trim();
  if (normalizedWorkingDir) {
    return normalizedWorkingDir;
  }

  if (!selectedRepository) {
    return DEFAULT_WORKING_DIR;
  }

  const parts = selectedRepository.split("/");
  const repoName = parts[parts.length - 1];

  return `${DEFAULT_WORKING_DIR}/${repoName}`;
}

export const CLOUD_WORKSPACE_ROOT = "/workspace";

/**
 * Absolute root the cloud Files tab lists and reads files against.
 *
 * With no repository and no known working dir, the agent may have written
 * anywhere under the sandbox workspace (e.g. cloned straight into
 * `/workspace/<repo>` instead of `/workspace/project`), so use the whole
 * workspace root. Otherwise anchor to the working dir / repo clone as
 * `getGitPath` does. File reads must use the same root as the listing, since
 * listed paths are relative to it.
 */
export function getCloudWorkspaceRoot(
  selectedRepository: string | null | undefined,
  workingDir?: string | null,
): string {
  if (!selectedRepository && !workingDir?.trim()) {
    return CLOUD_WORKSPACE_ROOT;
  }
  const gitPath = getGitPath(selectedRepository, workingDir);
  return gitPath.startsWith("/") ? gitPath : `/${gitPath}`;
}
