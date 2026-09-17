const GITHUB_TREE_URL =
  /^https?:\/\/(?:www\.)?github\.com\/([^/]+)\/([^/]+?)(?:\.git)?\/tree\/([^/]+)(?:\/(.+?))?\/?$/;

/**
 * Split a browser GitHub URL (`https://github.com/o/r/tree/<ref>/<path>`) into
 * a cloneable source, ref and repo path. A ref containing `/` is ambiguous in
 * this URL form, so only its first segment is taken as the ref.
 */
export function parseGitHubTreeUrl(
  url: string,
): { source: string; ref: string; repoPath: string | null } | null {
  const match = GITHUB_TREE_URL.exec(url.trim());
  if (!match) return null;
  const [, owner, repo, ref, repoPath] = match;
  return {
    source: `https://github.com/${owner}/${repo}`,
    ref,
    repoPath: repoPath ?? null,
  };
}
