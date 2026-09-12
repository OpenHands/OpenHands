import { Provider } from "#/types/settings";

export interface ParsedGitRemoteUrl {
  /** Original URL, trimmed. */
  url: string;
  /**
   * Browsable hostname of the remote (e.g. `github.com`, `git.example.com`).
   * Hosts that only serve SSH are mapped to their web equivalent, so callers
   * can build links from this directly. `url` keeps the original remote.
   */
  host: string | null;
  /** Path-style identifier, normalized to `owner/repo` (no leading slash, no `.git` suffix). */
  repository: string | null;
  /** Best-effort provider detection from the host. `null` for unrecognized/self-hosted hosts. */
  provider: Provider | null;
}

const KNOWN_HOSTS: Record<string, Provider> = {
  "github.com": "github",
  "gitlab.com": "gitlab",
  "bitbucket.org": "bitbucket",
  "dev.azure.com": "azure_devops",
};

// Hosts that serve git over SSH only, mapped to the host that serves the same
// repositories over the web. Azure DevOps hands out `ssh.dev.azure.com` (and
// `vs-ssh.visualstudio.com` for legacy organizations) in its SSH clone URLs,
// neither of which resolves a browsable page.
const SSH_WEB_HOSTS: Record<string, string> = {
  "ssh.dev.azure.com": "dev.azure.com",
  "vs-ssh.visualstudio.com": "dev.azure.com",
};

function toWebHost(host: string): string {
  return SSH_WEB_HOSTS[host.toLowerCase()] ?? host;
}

function stripGitSuffix(path: string): string {
  return path.replace(/\.git$/, "");
}

function detectProvider(host: string | null): Provider | null {
  if (!host) return null;
  return KNOWN_HOSTS[host.toLowerCase()] ?? null;
}

function normalizeAzureDevOpsPath(path: string): string {
  // Azure paths look like `org/project/_git/repo` or `org/_git/repo` over
  // HTTPS, and `v3/org/project/repo` over SSH. Normalize to `org/project/repo`
  // (or `org/repo`) so it lines up with constructBranchUrl's expectations.
  const segments = path.split("/").filter(Boolean);
  if (segments[0] === "v3") segments.shift();
  const gitIndex = segments.indexOf("_git");
  if (gitIndex === -1) return segments.join("/");
  return [...segments.slice(0, gitIndex), ...segments.slice(gitIndex + 1)].join(
    "/",
  );
}

function buildParsedGitRemoteUrl(
  url: string,
  rawHost: string | null,
  rawPath: string,
): ParsedGitRemoteUrl {
  const host = rawHost ? toWebHost(rawHost) : rawHost;
  const path = stripGitSuffix(rawPath.replace(/^\/+|\/+$/g, ""));
  const provider = detectProvider(host);
  const repository =
    provider === "azure_devops" ? normalizeAzureDevOpsPath(path) : path;

  return {
    url,
    host,
    repository: repository || null,
    provider,
  };
}

/**
 * Parse a git remote URL (HTTPS, SSH, or `git@host:path` shorthand) into its
 * host, repository (`owner/repo`), and best-effort provider.
 *
 * Returns `null` if the URL is empty or unparseable. Unknown hosts still
 * resolve `host` and `repository`; only `provider` is left `null`.
 */
export function parseGitRemoteUrl(
  remoteUrl: string | null | undefined,
): ParsedGitRemoteUrl | null {
  const url = remoteUrl?.trim();
  if (!url) return null;

  // ssh://, https://, http://, git://. Tried before the shorthand match
  // below, whose `[^@\s]+` would otherwise swallow `ssh://git` and read the
  // port in `ssh://git@host:2222/owner/repo` as the first path segment.
  try {
    const parsed = new URL(url);
    // A bare Windows path (`C:/src/repo`) parses as a `c:` scheme with an
    // empty host, so it is left to the shorthand match below.
    if (parsed.hostname) {
      return buildParsedGitRemoteUrl(url, parsed.hostname, parsed.pathname);
    }
  } catch {
    // Not a URL the WHATWG parser accepts, so try the shorthand form.
  }

  // git@host:owner/repo(.git)
  const scpMatch = url.match(/^[^@\s]+@([^:\s]+):(.+)$/);
  if (scpMatch) {
    const host = scpMatch[1];
    return buildParsedGitRemoteUrl(url, host, scpMatch[2]);
  }

  return null;
}
