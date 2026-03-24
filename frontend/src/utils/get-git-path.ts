/**
 * Get the git repository path for a conversation
 *
 * When sandbox grouping is enabled (strategy != NO_GROUPING), each conversation
 * gets its own subdirectory: /workspace/project/{conversationId}[/{repoName}]
 *
 * When sandbox grouping is disabled (NO_GROUPING), the path is simply:
 * /workspace/project[/{repoName}]
 *
 * @param conversationId The conversation ID
 * @param selectedRepository The selected repository (e.g., "OpenHands/OpenHands", "owner/repo", or "group/subgroup/repo")
 * @param useSandboxGrouping Whether sandbox grouping is enabled (strategy != NO_GROUPING)
 * @returns The git path to use
 */
export function getGitPath(
  conversationId: string,
  selectedRepository: string | null | undefined,
  useSandboxGrouping: boolean = false,
): string {
  // Base path depends on sandbox grouping strategy
  const basePath = useSandboxGrouping
    ? `/workspace/project/${conversationId}`
    : "/workspace/project";

  if (!selectedRepository) {
    return basePath;
  }

  // Extract the repository name from the path
  // The folder name is always the last part (handles both "owner/repo" and "group/subgroup/repo" formats)
  const parts = selectedRepository.split("/");
  const repoName = parts[parts.length - 1];

  return `${basePath}/${repoName}`;
}
