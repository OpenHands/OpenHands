/**
 * Get the git repository path for a conversation
 * If a repository is selected, returns /workspace/project/{repo-name}
 * Otherwise, returns /workspace/project
 *
 * @param selectedRepository The selected repository (e.g., "OpenHands/OpenHands", "owner/repo", or "group/subgroup/repo")
 * @returns The git path to use
 */
export function getGitPath(
  selectedRepository: string | null | undefined,
): string {
  if (!selectedRepository) {
    return "/workspace/project";
  }

  // Extract the repository name from the path
  // The folder name is always the last part (handles both "owner/repo" and "group/subgroup/repo" formats)
  const parts = selectedRepository.split("/");
  const repoName = parts[parts.length - 1];

  return `/workspace/project/${repoName}`;
}
