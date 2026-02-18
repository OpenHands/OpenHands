/**
 * Get the git repository path for a conversation.
 * Repos are cloned directly into /workspace/project, so the path is always the same.
 */
export function getGitPath(): string {
  return "/workspace/project";
}
