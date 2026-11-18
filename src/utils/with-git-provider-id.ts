import type { GitRepository } from "#/types/git";

/**
 * The repository stamped with the connected git provider it was picked from,
 * or with no id when the picker has none. Replaces an id left over from an
 * earlier selection, such as a stored recent repository.
 */
export const withGitProviderId = (
  repository: GitRepository,
  gitProviderId: number | null | undefined,
): GitRepository => {
  const { git_provider_id: _previous, ...rest } = repository;
  return gitProviderId == null
    ? rest
    : { ...rest, git_provider_id: gitProviderId };
};
