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
