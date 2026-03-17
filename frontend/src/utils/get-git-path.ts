/**
 * Get the git repository path for a conversation
 * If a repository is selected, returns {conversationId}/{repo-name}
 * Otherwise, returns {conversationId}
 *
 * Returns a relative path (not absolute) to avoid path duplication bug (#13327).
 * The agent server runs with working directory /workspace/project, so we only
 * need to provide the relative path from there.
 *
 * @param selectedRepository The selected repository (e.g., "OpenHands/OpenHands", "owner/repo", or "group/subgroup/repo")
 * @returns The relative git path from the working directory
 */
export function getGitPath(
  conversationId: string,
  selectedRepository: string | null | undefined,
): string {
  if (!selectedRepository) {
    return conversationId;
  }

  // Extract the repository name from the path
  // The folder name is always the last part (handles both "owner/repo" and "group/subgroup/repo" formats)
  const parts = selectedRepository.split("/");
  const repoName = parts[parts.length - 1];

  return `${conversationId}/${repoName}`;
}
