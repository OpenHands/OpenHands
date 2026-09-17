// Browser folder URL layouts, matched on the path so self-hosted instances
// work too. Order matters: Gitea's `src/branch/<ref>` would otherwise read
// `branch` as the ref.
const TREE_URL_PATTERNS = [
  // GitLab (incl. subgroups): <group…>/<repo>/-/tree/<ref>/<path>
  /^(?<repo>.+?)\/-\/tree\/(?<ref>[^/]+)(?:\/(?<path>.+))?$/,
  // Gitea / Forgejo: <o>/<r>/src/branch|tag|commit/<ref>/<path>
  /^(?<repo>[^/]+\/[^/]+)\/src\/(?:branch|tag|commit)\/(?<ref>[^/]+)(?:\/(?<path>.+))?$/,
  // GitHub: <o>/<r>/tree/<ref>/<path>; Bitbucket: <ws>/<r>/src/<ref>/<path>
  /^(?<repo>[^/]+\/[^/]+)\/(?:tree|src)\/(?<ref>[^/]+)(?:\/(?<path>.+))?$/,
];

/**
 * Split a git host's browser folder URL into a cloneable source, ref and repo
 * path. A ref containing `/` is ambiguous in these URLs, so only its first
 * segment is taken as the ref.
 */
export function parseGitTreeUrl(
  url: string,
): { source: string; ref: string; repoPath: string | null } | null {
  let parsed: URL;
  try {
    parsed = new URL(url.trim());
  } catch {
    return null;
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return null;

  const pathname = parsed.pathname.replace(/^\/+|\/+$/g, "");
  for (const pattern of TREE_URL_PATTERNS) {
    const groups = pattern.exec(pathname)?.groups;
    if (groups) {
      return {
        source: `${parsed.origin}/${groups.repo.replace(/\.git$/, "")}`,
        ref: decodeURIComponent(groups.ref),
        repoPath: groups.path ? decodeURIComponent(groups.path) : null,
      };
    }
  }
  return null;
}
