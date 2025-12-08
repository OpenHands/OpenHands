/**
 * Get the git repository path for a conversation
 * If a repository is selected, returns /workspace/project/{repo-name}
 * Otherwise, returns /workspace/project
 *
 * @param selectedRepository The selected repository (e.g., "OpenHands/OpenHands" or "owner/repo")
 * @returns The git path to use
 */
export function getGitPath(
  selectedRepository: string | null | undefined,
): string {
  if (!selectedRepository) {
    return "/workspace/project";
  }

  // Extract the repository name from "owner/repo" format
  // The folder name is the second part after "/"
  const parts = selectedRepository.split("/");
  const repoName = parts.length > 1 ? parts[1] : parts[0];

  return `/workspace/project/${repoName}`;
}
